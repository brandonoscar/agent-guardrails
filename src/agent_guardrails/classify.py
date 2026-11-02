"""Fail-closed read/write classification of tool names.

Every guard in this package first asks one question: is this tool a read?
Wrong in one direction, a write runs without approval. Wrong in the other, a
read waits behind an approval card nobody answers, and the run stalls for the
whole approval timeout. Over-gating a read is not "safe but noisy"; it is a
different outage.

The rule:

* The default verdict is WRITE. An unknown or new tool name is gated.
* A name is a READ only when its FIRST word is a read verb, no word is a
  write verb, and no word is a connective (``and``, ``or``, ``then``...).
  ``list_and_pay_invoices``, ``get_or_create_folder`` and ``run_query`` are
  writes: a read verb somewhere in a name proves nothing about the rest of it.
* ``known_reads`` lists reads the rule can't see (``weather_now``). A write
  verb still wins over a listing.
* Verbs are matched as whole words of the name, never as substrings. Substring
  matching fires ``SET`` inside ``SETTINGS``, ``ADD`` inside ``ADDRESSES`` and
  ``SHARE`` inside ``SHARED``, and gates reads such as ``get_settings`` or
  ``list_shared_drives``.

Namespaces. Only the tool's own name is classified: whatever comes before the
last ``.``, ``/``, ``:`` or ``__`` is a namespace (``crm.contacts.search``,
``github__list_issues``). A prefix joined with a single ``_`` or ``-``
(``GMAIL_FETCH_EMAILS``) can't be told apart from a verb, so declare it in
``namespaces``; until then such a name is gated.

There is deliberately no primary write denylist: a denylist fails open on
every name nobody thought to add. The write verbs only ever move a name from
read to write.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

#: Verbs that make a tool a read, when no write verb is present.
DEFAULT_READ_VERBS: frozenset[str] = frozenset(
    {
        "LIST",
        "GET",
        "SEARCH",
        "FIND",
        "FETCH",
        "READ",
        "DOWNLOAD",
        # Parse extracts a document's text without changing it. EXPORT is not
        # here: an export can send data somewhere else (``export_to_s3``).
        "PARSE",
        "QUERY",
        "LOOKUP",
        "AGGREGATE",
        "COUNT",
        "DESCRIBE",
    }
)

#: Verbs that make a tool a write, whatever else its name says.
#:
#: Verbs only. Words that are also common nouns in read tool names (LABEL,
#: DRAFT, WATCH, POST, CHARGE) are left out on purpose: as write verbs they
#: would gate reads like ``list_labels`` or ``get_post``. A name with no read
#: verb is gated anyway, so this list only matters for composite names such as
#: ``get_or_create_folder``.
DEFAULT_WRITE_VERBS: frozenset[str] = frozenset(
    {
        "CREATE",
        "UPDATE",
        "MODIFY",
        "PATCH",
        "PUT",
        "DELETE",
        "REMOVE",
        "SEND",
        "REPLY",
        "ADD",
        "MOVE",
        "TRASH",
        "ARCHIVE",
        "SHARE",
        "UPLOAD",
        "MARK",
        "SET",
        "ENABLE",
        "DISABLE",
        "CANCEL",
        "ACCEPT",
        "DECLINE",
        "INSERT",
        "IMPORT",
        "COPY",
        "RENAME",
        "EMPTY",
        "RESTORE",
        "UNTRASH",
        "REVOKE",
        "GRANT",
        "FORWARD",
        "SUBMIT",
        "EDIT",
        "APPEND",
        "CLEAR",
        "DUPLICATE",
        "RESET",
        "ASSIGN",
        "WRITE",
        "EXECUTE",
        "PUBLISH",
        "INVITE",
        "APPROVE",
        "PURGE",
        "DESTROY",
        "EXPORT",
        "UPSERT",
        "REPLACE",
        "MERGE",
    }
)

#: Words that join two actions in one name. A name that does two things is
#: classified by the riskier one, and the classifier can't see which that is.
DEFAULT_CONNECTIVES: frozenset[str] = frozenset({"AND", "OR", "THEN", "PLUS", "ELSE"})

_NON_ALNUM = re.compile(r"[^0-9A-Za-z]+")
# Explicit namespace separators: crm.contacts.search, github__list_issues,
# drive/list_files, server:get_status.
_NAMESPACE_SEP = re.compile(r"\.|/|:|__")
# Split camelCase and acronym boundaries: listInvoices -> list|Invoices,
# HTTPRequest -> HTTP|Request.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def words(name: str) -> tuple[str, ...]:
    """The whole words of a tool name, upper-cased, in order.

    ``send_email``, ``SEND_EMAIL``, ``sendEmail`` and ``send-email`` all give
    ``("SEND", "EMAIL")``.
    """
    out: list[str] = []
    for chunk in _NON_ALNUM.split(name):
        if chunk:
            out.extend(part for part in _CAMEL.split(chunk) if part)
    return tuple(word.upper() for word in out)


def segments(name: str) -> frozenset[str]:
    """The set of :func:`words` in a tool name."""
    return frozenset(words(name))


def own_name(name: str) -> str:
    """The tool's own name: what follows the last ``.``, ``/``, ``:`` or ``__``."""
    parts = [part for part in _NAMESPACE_SEP.split(name) if part.strip()]
    return parts[-1] if parts else ""


def _upper_all(values: Iterable[str]) -> frozenset[str]:
    return frozenset(value.upper() for value in values)


@dataclass(frozen=True)
class ToolClassifier:
    """Decides whether a tool name is a write. Fails closed.

    ``known_reads`` lists whole tool names that are reads although the rule
    can't prove it (``weather_now``). It is checked after the write verbs, so
    a known read that carries a write verb is still gated. ``namespaces``
    lists toolkit prefixes joined with ``_`` or ``-`` (``GMAIL`` in
    ``GMAIL_FETCH_EMAILS``) that are skipped before the first word is read.
    """

    read_verbs: frozenset[str] = DEFAULT_READ_VERBS
    write_verbs: frozenset[str] = DEFAULT_WRITE_VERBS
    known_reads: frozenset[str] = field(default_factory=frozenset)
    namespaces: frozenset[str] = field(default_factory=frozenset)
    connectives: frozenset[str] = DEFAULT_CONNECTIVES

    def __post_init__(self) -> None:
        object.__setattr__(self, "read_verbs", _upper_all(self.read_verbs))
        object.__setattr__(self, "write_verbs", _upper_all(self.write_verbs))
        object.__setattr__(self, "known_reads", _upper_all(self.known_reads))
        object.__setattr__(self, "namespaces", _upper_all(self.namespaces))
        object.__setattr__(self, "connectives", _upper_all(self.connectives))

    def is_write(self, name: str) -> bool:
        """True (gated) unless the name is provably a read."""
        if not name or not name.strip():
            return True
        own = own_name(name)
        found = list(words(own))
        if set(found) & self.write_verbs:
            return True
        if name.upper() in self.known_reads or own.upper() in self.known_reads:
            return False
        if set(found) & self.connectives:
            return True
        while found and found[0] in self.namespaces:
            found.pop(0)
        # Only a read verb in FIRST position proves a read. A name that starts
        # with anything else (run_query, do_the_thing) is unknown: gated.
        return not (found and found[0] in self.read_verbs)

    def is_read(self, name: str) -> bool:
        return not self.is_write(name)

    def extend(
        self,
        *,
        read_verbs: Iterable[str] = (),
        write_verbs: Iterable[str] = (),
        known_reads: Iterable[str] = (),
        namespaces: Iterable[str] = (),
        connectives: Iterable[str] = (),
    ) -> ToolClassifier:
        """A NEW classifier with more verbs, known reads or namespaces added.

        It changes nothing in place: pass the returned classifier to every
        guard that classifies (``gate_tools(..., classifier=c)`` and
        ``BudgetPolicy(classifier=c)``) so they agree.
        """
        return ToolClassifier(
            read_verbs=self.read_verbs | _upper_all(read_verbs),
            write_verbs=self.write_verbs | _upper_all(write_verbs),
            known_reads=self.known_reads | _upper_all(known_reads),
            namespaces=self.namespaces | _upper_all(namespaces),
            connectives=self.connectives | _upper_all(connectives),
        )


DEFAULT_CLASSIFIER = ToolClassifier()


def is_write_tool(name: str, classifier: ToolClassifier = DEFAULT_CLASSIFIER) -> bool:
    """Fail-closed verdict for one tool name: True means gate it."""
    return classifier.is_write(name)


__all__ = [
    "DEFAULT_CLASSIFIER",
    "DEFAULT_CONNECTIVES",
    "DEFAULT_READ_VERBS",
    "DEFAULT_WRITE_VERBS",
    "ToolClassifier",
    "is_write_tool",
    "own_name",
    "segments",
    "words",
]
