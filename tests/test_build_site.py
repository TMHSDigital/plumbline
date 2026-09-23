"""The site's navigation, as ``scripts/build_site.py`` writes it.

The script is not part of the package, so it is loaded from its path. These
cover the pure functions that write the header, the docs sidebar, the contents
list, and previous and next; ``scripts/check_site_links.mjs`` then checks every
link they produce resolves, on the assembled site, in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_site.py"


@pytest.fixture(scope="module")
def site() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_site", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_site"] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def test_every_doc_is_in_a_known_group_and_groups_are_contiguous(site: ModuleType) -> None:
    groups = [doc.group for doc in site.DOCS]
    assert set(groups) <= set(site.GROUPS)
    # Collapsing runs of the same group gives each group once, in GROUPS order,
    # so previous and next walk the sidebar top to bottom.
    runs = [g for i, g in enumerate(groups) if i == 0 or groups[i - 1] != g]
    assert runs == list(site.GROUPS)


def test_sidebar_marks_exactly_the_current_doc(site: ModuleType) -> None:
    html = site.docs_sidebar("methodology")
    assert html.count('aria-current="page"') == 1
    assert '<a href="methodology.html" aria-current="page">Methodology</a>' in html
    assert site.docs_sidebar(None).count("aria-current") == 0


def test_sidebar_links_absolutely_on_the_404_page(site: ModuleType) -> None:
    html = site.docs_sidebar(None, "/plumbline/docs/")
    assert 'href="/plumbline/docs/readme.html"' in html
    assert 'href="/plumbline/docs/"' in html


def test_prev_next_at_the_ends_and_in_the_middle(site: ModuleType) -> None:
    first, second, last = site.DOCS[0], site.DOCS[1], site.DOCS[-1]
    start = site.prev_next(first.slug)
    assert 'class="prev"' not in start and f'href="{second.slug}.html"' in start
    end = site.prev_next(last.slug)
    assert 'class="next"' not in end and f'href="{site.DOCS[-2].slug}.html"' in end
    middle = site.prev_next(second.slug)
    assert f'href="{first.slug}.html"' in middle and f'href="{site.DOCS[2].slug}.html"' in middle


def heading(level: int, id: str) -> dict[str, object]:
    return {"level": level, "id": id, "text": id.title()}


def test_toc_nests_h3_under_its_h2_and_skips_other_levels(site: ModuleType) -> None:
    html = site.page_toc(
        [
            heading(1, "title"),
            heading(3, "orphan"),
            heading(2, "one"),
            heading(3, "one-a"),
            heading(4, "deep"),
            heading(2, "two"),
        ]
    )
    assert '<li><a href="#one">One</a><ul><li><a href="#one-a">One-A</a></li></ul></li>' in html
    assert '<li><a href="#two">Two</a></li>' in html
    for skipped in ("#title", "#orphan", "#deep"):
        assert skipped not in html


def test_toc_is_omitted_below_two_sections(site: ModuleType) -> None:
    assert site.page_toc([heading(1, "t"), heading(2, "only"), heading(3, "sub")]) == ""


def test_toc_escapes_heading_text(site: ModuleType) -> None:
    html = site.page_toc(
        [
            {"level": 2, "id": "a", "text": "<b>&"},
            {"level": 2, "id": "b", "text": "b"},
        ]
    )
    assert "&lt;b&gt;&amp;" in html


def test_header_marks_the_current_link(site: ModuleType) -> None:
    html = site.site_header("../", "docs")
    assert '<a href="../docs/" aria-current="page">Docs</a>' in html
    assert html.count("aria-current") == 1
    assert site.site_header("/plumbline/", None).count("aria-current") == 0
    # The controls that need a script start hidden.
    assert 'class="search-open" hidden' in html and 'class="theme-toggle" hidden' in html


def test_explainer_needs_exactly_one_header_slot(site: ModuleType) -> None:
    page = site.explainer_page(f"<body>{site.HEADER_SLOT}<main></main></body>")
    assert site.HEADER_SLOT not in page and 'class="site-header"' in page
    for broken in ("<body></body>", site.HEADER_SLOT * 2):
        with pytest.raises(site.BuildError):
            site.explainer_page(broken)


def test_the_committed_explainer_has_its_slot(site: ModuleType) -> None:
    source = (site.SITE / "index.html").read_text(encoding="utf-8")
    assert source.count(site.HEADER_SLOT) == 1


def fake_rendered(site: ModuleType) -> dict[str, dict[str, object]]:
    def sections(label: str) -> list[dict[str, object]]:
        return [
            {"id": None, "heading": "", "level": 0, "text": "before"},
            {"id": "top", "heading": label, "level": 1, "text": "lead"},
            {"id": "one", "heading": "One", "level": 2, "text": "body one"},
        ]

    return {
        doc.slug: {"html": "", "headings": [], "sections": sections(doc.label)} for doc in site.DOCS
    }


EXPLAINER = """<h1>A &amp; B</h1><p class="lede">Lede <em>text</em>.</p>
<section id="argument" aria-labelledby="a"><h2 id="a">The argument</h2><p>Zero is
not <b>reachable</b>.</p></section>"""


def test_search_index_lists_each_page_then_its_sections(site: ModuleType) -> None:
    index = site.search_index(fake_rendered(site), EXPLAINER)
    assert index[0] == {"t": site.EXPLAINER_LABEL, "h": "A & B", "u": "./", "x": "Lede text."}
    assert index[1]["u"] == "./#argument" and index[1]["x"] == "Zero is not reachable."
    first = site.DOCS[0]
    page = {"t": first.label, "h": first.label, "u": f"docs/{first.slug}.html", "x": "before lead"}
    assert index[2] == page
    assert index[3]["u"] == f"docs/{first.slug}.html#one" and index[3]["x"] == "body one"
    assert len(index) == 2 + 2 * len(site.DOCS)


def test_search_index_refuses_an_explainer_it_cannot_read(site: ModuleType) -> None:
    with pytest.raises(site.BuildError):
        site.search_index(fake_rendered(site), "<p>no heading</p>")
    broken = EXPLAINER.replace('<h2 id="a">The argument</h2>', "")
    with pytest.raises(site.BuildError):
        site.search_index(fake_rendered(site), broken)


def test_search_index_caps_long_sections(site: ModuleType) -> None:
    rendered = fake_rendered(site)
    rendered[site.DOCS[0].slug]["sections"][2]["text"] = "x" * 10_000
    index = site.search_index(rendered, EXPLAINER)
    assert max(len(entry["x"]) for entry in index) == site.SEARCH_TEXT_LIMIT
