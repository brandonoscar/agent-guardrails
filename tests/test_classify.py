from __future__ import annotations

import pytest

from agent_guardrails.classify import (
    DEFAULT_CLASSIFIER,
    ToolClassifier,
    is_write_tool,
    own_name,
    segments,
    words,
)


@pytest.mark.parametrize(
    "name",
    [
        "send_email",
        "create_invoice",
        "delete_user",
        "GMAIL_SEND_EMAIL",
        "get_or_create_folder",  # a read verb does not rescue a write verb
        "search_and_delete",
        "mark_as_read",
        "do_the_thing",  # no verb at all: unknown, so gated
        "frobnicate",
        "",
        "   ",
        # A read verb somewhere in the name proves nothing about the rest of it.
        "list_and_pay_invoices",
        "query_and_transfer_funds",
        "lookup_and_refund",
        "find_and_replace",
        "get_or_upsert_contact",
        "fetch_then_archive",
        "run_query",  # the first word decides, and RUN is not a read verb
        "contacts_search",  # a read verb that isn't first proves nothing
        "export_to_s3",  # an export can send data somewhere else
        "export_document",
        "GMAIL_FETCH_EMAILS",  # an undeclared prefix can't be told from a verb
    ],
)
def test_writes_and_unknowns_are_gated(name: str) -> None:
    assert is_write_tool(name)


@pytest.mark.parametrize(
    "name",
    [
        "get_weather",
        "list_invoices",
        "search_contacts",
        "listInvoices",
        "fetch-orders",
        "crm.contacts.search",  # an explicit namespace is skipped
        "github__list_issues",
        "drive/list_files",
        "count_open_tickets",
        "parse_document",
    ],
)
def test_provable_reads_pass(name: str) -> None:
    assert not is_write_tool(name)


@pytest.mark.parametrize(
    "name",
    [
        "get_settings",  # SET is not inside SETTINGS
        "list_addresses",  # ADD is not inside ADDRESSES
        "list_shared_drives",  # SHARE is not inside SHARED
        "get_forwarding_rules",  # FORWARD is not inside FORWARDING
        "list_labels",  # nouns that look like verbs are not write verbs
        "get_post",
    ],
)
def test_verbs_match_whole_words_only(name: str) -> None:
    assert not is_write_tool(name)


def test_declared_namespaces_are_skipped_before_the_first_word() -> None:
    classifier = DEFAULT_CLASSIFIER.extend(namespaces=["gmail", "GOOGLEDRIVE"])
    assert classifier.is_read("GMAIL_FETCH_EMAILS")
    assert classifier.is_read("GOOGLEDRIVE_LIST_FILES")
    assert classifier.is_write("GMAIL_SEND_EMAIL")
    assert classifier.is_write("GMAIL_FETCH_AND_DELETE_EMAILS")
    assert classifier.is_write("GMAIL")  # nothing left after the namespace


def test_extend_returns_a_new_classifier_and_changes_nothing_in_place() -> None:
    extended = DEFAULT_CLASSIFIER.extend(known_reads=["weather_now"])
    assert extended.is_read("weather_now")
    assert DEFAULT_CLASSIFIER.is_write("weather_now")
    assert extended is not DEFAULT_CLASSIFIER


def test_words_keep_their_order_and_own_name_drops_namespaces() -> None:
    assert words("listInvoices") == ("LIST", "INVOICES")
    assert words("GMAIL_FETCH_EMAILS") == ("GMAIL", "FETCH", "EMAILS")
    assert own_name("crm.contacts.search") == "search"
    assert own_name("mcp__github__list_issues") == "list_issues"
    assert own_name("send_email") == "send_email"


def test_segments_split_snake_kebab_dots_and_camel() -> None:
    assert segments("send_email") == {"SEND", "EMAIL"}
    assert segments("sendEmail") == {"SEND", "EMAIL"}
    assert segments("send-email") == {"SEND", "EMAIL"}
    assert segments("crm.contacts.search") == {"CRM", "CONTACTS", "SEARCH"}
    assert segments("HTTPRequestGet") == {"HTTP", "REQUEST", "GET"}


def test_known_reads_need_no_verb_but_lose_to_a_write_verb() -> None:
    classifier = ToolClassifier(
        known_reads=frozenset({"weather_now", "send_status", "get_user_and_orders"})
    )
    assert classifier.is_read("weather_now")
    assert classifier.is_read("WEATHER_NOW")
    # A listed read may join two reads; the listing is the proof.
    assert classifier.is_read("get_user_and_orders")
    # A known read that carries a write verb stays gated: fail closed.
    assert classifier.is_write("send_status")


def test_extend_adds_verbs_without_losing_defaults() -> None:
    classifier = DEFAULT_CLASSIFIER.extend(read_verbs=["inspect"], write_verbs=["refund"])
    assert classifier.is_read("inspect_order")
    assert classifier.is_write("get_and_refund_order")
    assert classifier.is_read("get_order")
    assert classifier.is_write("send_order")


def test_lowercase_configuration_is_normalized() -> None:
    classifier = ToolClassifier(read_verbs=frozenset({"peek"}), write_verbs=frozenset({"poke"}))
    assert classifier.is_read("peek_value")
    assert classifier.is_write("peek_and_poke")
    # Custom verb sets replace the defaults entirely.
    assert classifier.is_write("get_value")
