from podcastpipe.models import Story
from podcastpipe.stages.research import (
    FEED_AGENT,
    _looks_like_html,
    dedupe_stories,
    is_recent,
    normalize_url,
    score_story,
    title_key,
    titles_match,
)


def story(title, url, summary="", score=0.0, source="src"):
    return Story(title=title, url=url, source=source, summary=summary, score=score)


class TestUrlNormalization:
    def test_strips_tracking_params(self):
        assert (
            normalize_url("https://x.com/a?utm_source=rss&utm_medium=feed")
            == "https://x.com/a"
        )

    def test_strips_fragment_and_trailing_slash(self):
        assert normalize_url("https://x.com/a/#top") == "https://x.com/a"

    def test_keeps_meaningful_query(self):
        assert normalize_url("https://x.com/a?id=7") == "https://x.com/a?id=7"

    def test_empty(self):
        assert normalize_url("") == ""


class TestTitleMatching:
    def test_stopwords_removed(self):
        assert "the" not in title_key("The vote on the budget").split()

    def test_identical_titles_match(self):
        assert titles_match("Council approves budget", "Council approves budget")

    def test_reworded_headline_matches(self):
        assert titles_match(
            "Town council approves the 2026 budget",
            "Town Council Approves 2026 Budget",
        )

    def test_different_stories_do_not_match(self):
        assert not titles_match(
            "Council approves budget", "Fire crews respond to Franklin Street blaze"
        )

    def test_empty_titles_never_match(self):
        assert not titles_match("", "")


class TestDedupe:
    def test_same_url_collapses(self):
        stories = [
            story("A", "https://x.com/a", score=1.0),
            story("A different headline", "https://x.com/a?utm_source=rss", score=5.0),
        ]
        assert len(dedupe_stories(stories)) == 1

    def test_keeps_highest_scoring_duplicate(self):
        stories = [
            story("Council approves budget", "https://a.com/1", score=1.0),
            story("Council Approves Budget", "https://b.com/2", score=9.0),
        ]
        kept = dedupe_stories(stories)
        assert len(kept) == 1
        assert kept[0].url == "https://b.com/2"

    def test_seen_urls_are_dropped(self):
        stories = [story("A", "https://x.com/a"), story("B", "https://x.com/b")]
        kept = dedupe_stories(stories, seen_urls={"https://x.com/a"})
        assert [s.url for s in kept] == ["https://x.com/b"]

    def test_seen_urls_normalized_before_comparison(self):
        stories = [story("A", "https://x.com/a")]
        assert dedupe_stories(stories, seen_urls={"https://x.com/a/?utm_source=x"}) == []

    def test_stories_without_urls_are_dropped(self):
        assert dedupe_stories([story("A", "")]) == []

    def test_distinct_stories_survive(self):
        stories = [
            story("Council approves budget", "https://a.com/1"),
            story("Fire on Franklin Street", "https://a.com/2"),
            story("Agency expands eligibility", "https://a.com/3"),
        ]
        assert len(dedupe_stories(stories)) == 3


class TestScoring:
    keywords = {"springfield": 5.0, "resident": 4.0}

    def test_headline_hit_counts_double(self):
        in_title = score_story(story("Springfield votes", "u"), self.keywords)
        in_body = score_story(story("Town votes", "u", "in springfield"), self.keywords)
        assert in_title == in_body * 2

    def test_multiple_keywords_accumulate(self):
        both = score_story(story("Springfield resident honored", "u"), self.keywords)
        assert both == (5.0 + 4.0) * 2

    def test_source_weight_multiplies(self):
        base = score_story(story("Springfield news", "u"), self.keywords, 1.0)
        weighted = score_story(story("Springfield news", "u"), self.keywords, 2.0)
        assert weighted == base * 2

    def test_no_match_scores_zero(self):
        assert score_story(story("Unrelated", "u"), self.keywords) == 0.0


class TestRecency:
    def test_missing_date_is_kept(self):
        # Small-town feeds omit dates; dropping them loses local coverage.
        assert is_recent("", 24)

    def test_unparseable_date_is_kept(self):
        assert is_recent("last Tuesday", 24)

    def test_old_iso_date_rejected(self):
        assert not is_recent("2020-01-01T00:00:00+00:00", 24)

    def test_rfc2822_parsed(self):
        assert not is_recent("Wed, 01 Jan 2020 00:00:00 +0000", 24)


class TestFeedFetching:
    """A blocked request and a malformed feed look identical to the XML parser.

    Telling them apart is the whole point of fetching the bytes ourselves:
    "unreadable feed" sent a week of work into replacing Military.com and Stars
    and Stripes URLs that were never broken, when the servers were answering a
    non-browser user-agent with a challenge page.
    """

    def test_html_block_page_detected_by_content_type(self):
        assert _looks_like_html(b"whatever", "text/html; charset=utf-8")

    def test_html_block_page_detected_by_body(self):
        assert _looks_like_html(b"\n  <!DOCTYPE html><html>", "")
        assert _looks_like_html(b"<html><head>", "application/octet-stream")

    def test_real_feed_is_not_html(self):
        assert not _looks_like_html(b"<?xml version='1.0'?><rss>", "application/rss+xml")

    def test_xml_served_as_html_content_type_is_still_flagged(self):
        # Publishers that block bots frequently answer with text/html even when
        # the body is empty; flagging on either signal is deliberate.
        assert _looks_like_html(b"", "text/html")

    def test_browser_user_agent_is_sent(self):
        # The default feedparser agent is what gets blocked. If this ever
        # regresses to something identifying as a bot, the feeds go quiet again
        # and the only symptom is a shorter brief.
        assert "Mozilla/5.0" in FEED_AGENT
        assert "feedparser" not in FEED_AGENT.lower()

    def test_fetch_failure_is_reported_not_raised(self, monkeypatch):
        # One dead feed must never stop the show.
        import podcastpipe.stages.research as research

        def boom(*args, **kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        parsed, error = research.fetch_feed("https://example.com/feed")
        assert parsed is None
        assert "connection refused" in error
