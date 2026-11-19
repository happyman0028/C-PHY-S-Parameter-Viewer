from widgets.trace_selector import wildcard_search_matches


SEARCHABLE = r"UL4U.3J04M DUT_V2 C:\measurements\4Y4EJ\sample.s6p Sxx21_1R"


def test_plain_search_is_case_insensitive_substring():
    assert wildcard_search_matches("ul4u.3j04m", SEARCHABLE)


def test_asterisk_wildcard_matches_across_characters():
    assert wildcard_search_matches("UL4U*J04M", SEARCHABLE)
    assert wildcard_search_matches("S*21_1R", SEARCHABLE)


def test_slash_is_or_and_space_is_and():
    assert wildcard_search_matches("missing / 4Y4EJ", SEARCHABLE)
    assert wildcard_search_matches("UL4U* Sxx*", SEARCHABLE)
    assert not wildcard_search_matches("UL4U* Syy*", SEARCHABLE)
