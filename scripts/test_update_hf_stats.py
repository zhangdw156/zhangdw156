import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import call, patch
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import update_hf_stats as stats


class DownloadStatsTests(unittest.TestCase):
    def test_sums_lifetime_counts_across_both_paginated_repo_types(self):
        def page(url):
            parsed = urlsplit(url)
            query = parse_qs(parsed.query)
            kind = parsed.path.rsplit("/", 1)[-1]
            if "cursor" not in query:
                self.assertEqual(query["author"], ["zhangdw"])
                self.assertEqual(query["expand[]"], ["downloadsAllTime"])
                count = 40 if kind == "models" else 1000
                next_url = f"https://huggingface.co/api/{kind}?author=zhangdw&cursor=next"
                links = f'<{url}>; rel="first", <{next_url}>; rel="next"'
                suffix = "first"
            else:
                count = 2 if kind == "models" else 3
                links = ""
                suffix = "second"
            return [{
                "id": f"zhangdw/{kind}-{suffix}",
                "downloads": 9999,
                "downloadsAllTime": count,
            }], links

        with patch.object(stats, "fetch_page", side_effect=page) as fetch:
            self.assertEqual(stats.total_downloads(), 1045)
        self.assertEqual(fetch.call_count, 4)

    def test_missing_or_invalid_lifetime_count_preserves_both_old_cards(self):
        invalid_counts = [None, "123", -1, True]
        for count in invalid_counts:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as temp:
                destination = Path(temp)
                previous = self.seed_cards(destination)
                dataset = {"id": "zhangdw/dataset", "downloads": 9999}
                if count is not None:
                    dataset["downloadsAllTime"] = count
                pages = [
                    ([{"id": "zhangdw/model", "downloadsAllTime": 10}], ""),
                    ([dataset], ""),
                ]
                with patch.object(stats, "fetch_page", side_effect=pages):
                    with self.assertRaises(ValueError):
                        stats.update_cards(destination)
                self.assert_cards_unchanged(destination, previous)

    def test_failure_after_partial_fetch_preserves_both_old_cards(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            previous = self.seed_cards(destination)
            pages = [
                ([{"id": "zhangdw/model", "downloadsAllTime": 10}], ""),
                URLError("Hub is unavailable"),
            ]
            with patch.object(stats, "fetch_page", side_effect=pages):
                with self.assertRaises(URLError):
                    stats.update_cards(destination)
            self.assert_cards_unchanged(destination, previous)

    def test_transient_http_and_network_failures_retry_then_succeed(self):
        response = io.BytesIO(json.dumps([{"downloadsAllTime": 123}]).encode())
        response.headers = {"Link": '<https://huggingface.co/api/models?cursor=next>; rel="next"'}
        http_error = HTTPError("https://huggingface.co/api/models", 503, "Unavailable", {}, None)
        self.addCleanup(http_error.close)
        attempts = [
            http_error,
            URLError("Temporary connection error"),
            response,
        ]
        with patch.object(stats, "urlopen", side_effect=attempts) as fetch:
            with patch.object(stats.time, "sleep") as sleep:
                data, link = stats.fetch_page("https://huggingface.co/api/models")
        self.assertEqual(data, [{"downloadsAllTime": 123}])
        self.assertIn('rel="next"', link)
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(1), call(2)])

    def test_retries_are_bounded_and_permanent_errors_fail_immediately(self):
        for status, expected_calls in [(429, 3), (503, 3), (404, 1)]:
            with self.subTest(status=status):
                error = HTTPError("https://huggingface.co/api/models", status, "Error", {}, None)
                self.addCleanup(error.close)
                with patch.object(stats, "urlopen", side_effect=error) as fetch:
                    with patch.object(stats.time, "sleep"):
                        with self.assertRaises(HTTPError):
                            stats.fetch_page("https://huggingface.co/api/models")
                self.assertEqual(fetch.call_count, expected_calls)

    def test_success_renders_the_same_single_total_in_both_themes(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp)
            with patch.object(stats, "total_downloads", return_value=123456) as total:
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(stats.update_cards(destination), 123456)
            total.assert_called_once_with()
            for filename in stats.PALETTES:
                svg = (destination / filename).read_text(encoding="utf-8")
                root = stats.ElementTree.fromstring(svg)
                ns = {"svg": "http://www.w3.org/2000/svg"}
                numbers = root.findall(".//svg:text[@class='number']", ns)
                self.assertEqual([node.text for node in numbers], ["123,456"])
                self.assertIn("HF 下载量", svg)

    def seed_cards(self, destination):
        previous = {name: f"previous card: {name}" for name in stats.PALETTES}
        for name, content in previous.items():
            (destination / name).write_text(content, encoding="utf-8")
        return previous

    def assert_cards_unchanged(self, destination, previous):
        for name, content in previous.items():
            self.assertEqual((destination / name).read_text(encoding="utf-8"), content)


if __name__ == "__main__":
    unittest.main()
