"""Opt-in Chromium desktop/mobile smoke checks with a synthetic camera, never a physical device."""

import json

from demo_server import demo_server
from playwright.sync_api import expect, sync_playwright


def main() -> None:
    with demo_server() as (url, token, output), sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"]
        )
        try:
            results = []
            for width, height in [(1440, 1000), (390, 844)]:
                context = browser.new_context(
                    viewport={"width": width, "height": height},
                    reduced_motion="reduce",
                    permissions=["camera"],
                    has_touch=width < 500,
                )
                page = context.new_page()
                errors = []
                requests = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda request: requests.append(request.url))
                page.goto(url)
                expect(page.get_by_role("heading", level=1)).to_have_text("Laboratório de Expressões Faciais")
                expect(page.locator("#toggle-stream")).to_be_disabled()
                page.locator("#api-token").fill(token)
                page.locator("#api-token").blur()
                page.locator("#session-consent").check()
                page.locator("#toggle-stream").click()
                expect(page.locator("#status-badge")).to_have_text("Online · vídeo", timeout=90_000)
                expect(page.locator("#track-count")).not_to_have_text("0", timeout=10_000)
                expect(page.locator("#expression-summary")).to_contain_text("neutral", timeout=10_000)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                    "Horizontal overflow"
                )
                with page.expect_download() as exported:
                    page.locator("#export-metrics").click()
                export_path = output / f"session-{width}.json"
                exported.value.save_as(export_path)
                assert json.loads(export_path.read_text())["mode"] == "server"
                page.screenshot(path=str(output / f"dashboard-{width}.png"), full_page=True)
                page.locator("#session-consent").uncheck()
                expect(page.locator("#status-badge")).to_have_text("Offline")
                expect(page.locator("#toggle-stream")).to_be_disabled()
                # The styled span covers the radio input, so click its label as a user would.
                page.locator('label:has(input[name="source-mode"][value="camera"])').click()
                page.locator("#session-consent").check()
                page.locator("#toggle-stream").click()
                expect(page.locator("#status-badge")).to_have_text("Online · câmera", timeout=60_000)
                expect(page.locator("#analysis-latency")).to_contain_text(
                    "Latência de análise:", timeout=20_000
                )
                page.locator("#session-consent").uncheck()
                expect(page.locator("#status-badge")).to_have_text("Offline")
                assert page.evaluate("document.getElementById('camera-video').srcObject === null"), (
                    "Camera was not released"
                )
                assert not errors, errors
                assert all(request.startswith(url) for request in requests), "External request in dashboard"
                assert not any("?token=" in request for request in requests)
                results.append(
                    {
                        "viewport": f"{width}x{height}",
                        "video": "passed",
                        "synthetic_camera": "passed",
                        "export": "passed",
                        "overflow": False,
                        "page_errors": errors,
                        "external_requests": 0,
                    }
                )
                context.close()
            (output / "ui-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(results))
        finally:
            browser.close()


if __name__ == "__main__":
    main()
