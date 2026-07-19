#!/usr/bin/env python3
"""
YouTube RSSから登録チャンネルの新着動画を取得し、
YouTube字幕を基に日本語要約を生成して data/videos.json を更新します。

処理の流れ:
1. YouTube RSSから動画情報を取得
2. 公開字幕・自動生成字幕を取得
3. OPENAI_API_KEYがあれば字幕をAI要約
4. 字幕が取れない場合は概要欄を要約
5. 過去に字幕要約済みの動画は再利用してAPI利用量を抑える
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import requests
from youtube_transcript_api import YouTubeTranscriptApi


ROOT = Path(__file__).resolve().parents[1]
CHANNELS_PATH = ROOT / "channels.json"
OUTPUT_PATH = ROOT / "data" / "videos.json"

MAX_VIDEOS_PER_CHANNEL = int(os.getenv("MAX_VIDEOS_PER_CHANNEL", "8"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "30000"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
RSS_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

TRANSCRIPT_LANGUAGES = ["ja", "en"]
YTT_API = YouTubeTranscriptApi()


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_existing_videos() -> dict[str, dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return {}

    try:
        payload = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        return {
            item["videoId"]: item
            for item in payload.get("videos", [])
            if item.get("videoId")
        }
    except Exception as exc:
        print(f"既存データを読み込めませんでした: {exc}", file=sys.stderr)
        return {}


def fetch_transcript(video_id: str) -> tuple[str, str]:
    """
    戻り値:
      (字幕本文, 状態)
      状態は ok / unavailable
    """
    errors: list[str] = []

    try:
        fetched = YTT_API.fetch(
            video_id,
            languages=TRANSCRIPT_LANGUAGES,
        )
        text = clean_text(" ".join(snippet.text for snippet in fetched))
        if text:
            return text[:MAX_TRANSCRIPT_CHARS], "ok"
    except Exception as exc:
        errors.append(str(exc))

    # 日本語・英語以外の字幕しかない場合は、利用可能な先頭字幕を試す
    try:
        transcript_list = YTT_API.list(video_id)
        transcript = next(iter(transcript_list))
        fetched = transcript.fetch()
        text = clean_text(" ".join(snippet.text for snippet in fetched))
        if text:
            return text[:MAX_TRANSCRIPT_CHARS], "ok"
    except Exception as exc:
        errors.append(str(exc))

    detail = " / ".join(error for error in errors if error)
    if detail:
        print(f"字幕取得失敗: {video_id}: {detail[:500]}", file=sys.stderr)
    else:
        print(f"字幕取得失敗: {video_id}: 字幕が空でした", file=sys.stderr)

    return "", "unavailable"


def fallback_summary(title: str, source_text: str) -> dict[str, Any]:
    source = clean_text(source_text)
    if source:
        summary = source[:240] + ("…" if len(source) > 240 else "")
    else:
        summary = (
            f"「{title}」に関する新着動画です。"
            "字幕と概要欄から内容を取得できなかったため、詳細はリンク先で確認してください。"
        )

    lowered = f"{title} {source_text}".lower()
    high_words = (
        "ai",
        "生成ai",
        "claude",
        "chatgpt",
        "資産",
        "金利",
        "為替",
        "日本経済",
    )
    priority = "high" if any(word in lowered for word in high_words) else "medium"

    tags: list[str] = []
    keyword_map = {
        "AI": ("ai", "chatgpt", "claude", "生成ai", "エージェント"),
        "資産管理": ("資産", "投資", "金", "株", "富裕層"),
        "国際情勢": ("ニュージーランド", "中国", "移民", "海外", "戦争"),
        "日本経済": ("日本", "金利", "為替", "財政", "産業"),
    }

    for label, words in keyword_map.items():
        if any(word in lowered for word in words):
            tags.append(label)

    return {
        "summary": summary,
        "priority": priority,
        "tags": tags[:4] or ["新着動画"],
    }


def extract_response_text(payload: dict[str, Any]) -> str:
    text = payload.get("output_text")
    if text:
        return str(text)

    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])

    raise ValueError("OpenAI APIから出力テキストを取得できませんでした。")


def call_openai(
    title: str,
    transcript: str,
    description: str,
    category: str,
) -> dict[str, Any]:
    source_kind = "字幕" if transcript else "概要欄"
    source_text = transcript or clean_text(description)

    if not OPENAI_API_KEY:
        return fallback_summary(title, source_text)

    prompt = f"""
次のYouTube動画について、{source_kind}の内容を基に、
忙しい人が動画を見なくても主要点を把握できる日本語要約を作成してください。

要約ルール:
- 動画の主張、根拠、結論を中心にまとめる
- 冒頭のあいさつ、宣伝、定型文、重複は除く
- 字幕にない事実を補わない
- 内容が断定的・推測的な場合は、その性質が分かる表現にする
- 140～240文字
- 「この動画では」で始めなくてよい
- 利用者の関心との関連度で優先度を判定する

利用者の関心:
自治体BPR、生成AI、国際情勢、資産管理、投資、仕事と暮らしの自動化

チャンネル分類:
{category}

タイトル:
{title}

情報源:
{source_kind}

内容:
{source_text}

JSONだけを返してください。
形式:
{{
  "summary": "140～240文字の要約",
  "priority": "high または medium または low",
  "tags": ["最大4個"]
}}
""".strip()

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "input": prompt,
        },
        timeout=120,
    )
    response.raise_for_status()

    text = extract_response_text(response.json()).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    data = json.loads(text)

    if data.get("priority") not in {"high", "medium", "low"}:
        data["priority"] = "medium"
    if not isinstance(data.get("tags"), list):
        data["tags"] = []

    return {
        "summary": str(data.get("summary", "")).strip(),
        "priority": data["priority"],
        "tags": [str(item) for item in data["tags"][:4]],
    }


def get_thumbnail(entry: Any, video_id: str) -> str:
    media_group = entry.get("media_group") or []
    if media_group:
        thumbnails = media_group[0].get("media_thumbnail") or []
        if thumbnails and thumbnails[0].get("url"):
            return thumbnails[0]["url"]

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def can_reuse_existing(item: dict[str, Any] | None) -> bool:
    if not item or not item.get("summary"):
        return False

    source = item.get("summarySource", "")

    # APIキーがある場合はAI字幕要約だけを再利用する。
    # APIキーを後から追加した場合、旧来の抜粋は自動的に作り直される。
    if OPENAI_API_KEY:
        return source == "transcript_ai"

    # APIキーがない場合は、字幕抜粋を再利用する。
    return source == "transcript_excerpt"


def update() -> None:
    channels = json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))
    existing_videos = load_existing_videos()

    videos: list[dict[str, Any]] = []
    errors: list[str] = []

    for channel in channels:
        channel_id = channel["channelId"]
        feed_url = RSS_TEMPLATE.format(channel_id=channel_id)
        feed = feedparser.parse(feed_url)

        if getattr(feed, "bozo", False) and not feed.entries:
            errors.append(f'{channel["name"]}: RSS取得失敗')
            continue

        for entry in feed.entries[:MAX_VIDEOS_PER_CHANNEL]:
            video_id = entry.get("yt_videoid") or entry.get("videoid")

            if not video_id:
                match = re.search(
                    r"(?:v=|/)([\w-]{11})(?:[?&/]|$)",
                    entry.get("link", ""),
                )
                video_id = match.group(1) if match else None

            if not video_id:
                continue

            title = clean_text(entry.get("title"))

            description = ""
            media_group = entry.get("media_group") or []
            if media_group:
                description = media_group[0].get("media_description", "")
            description = description or entry.get("summary", "")
            description = clean_text(description)

            existing = existing_videos.get(video_id)

            if can_reuse_existing(existing):
                ai = {
                    "summary": existing["summary"],
                    "priority": existing.get("priority", "medium"),
                    "tags": existing.get("tags", []),
                }
                transcript_status = existing.get("transcriptStatus", "ok")
                summary_source = existing.get("summarySource", "transcript_ai")
            else:
                transcript, transcript_status = fetch_transcript(video_id)

                try:
                    ai = call_openai(
                        title=title,
                        transcript=transcript,
                        description=description,
                        category=channel.get("category", ""),
                    )
                except Exception as exc:
                    print(f"要約生成をスキップ: {title}: {exc}", file=sys.stderr)
                    ai = fallback_summary(title, transcript or description)

                if transcript:
                    summary_source = (
                        "transcript_ai"
                        if OPENAI_API_KEY
                        else "transcript_excerpt"
                    )
                else:
                    summary_source = (
                        "description_ai"
                        if OPENAI_API_KEY
                        else "description_excerpt"
                    )

            videos.append(
                {
                    "videoId": video_id,
                    "channel": channel["name"],
                    "channelHandle": channel.get("handle", ""),
                    "title": title,
                    "published": entry.get("published") or entry.get("updated"),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": get_thumbnail(entry, video_id),
                    "description": description,
                    "summary": ai["summary"],
                    "priority": ai["priority"],
                    "tags": ai["tags"],
                    "summarySource": summary_source,
                    "transcriptStatus": transcript_status,
                    "durationMinutes": None,
                }
            )

    videos.sort(key=lambda item: item.get("published") or "", reverse=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "channelCount": len(channels),
                "videos": videos,
                "errors": errors,
                "message": "本日の新着なし" if not videos and not errors else "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"{len(videos)}件を {OUTPUT_PATH} に保存しました。")
    if errors:
        print("エラー: " + " / ".join(errors), file=sys.stderr)


if __name__ == "__main__":
    update()
