#!/usr/bin/env python3
"""
YouTube新着ダッシュボード：自宅PC・ローカルAI要約版

- YouTube字幕を自宅回線から取得
- Ollamaで動くローカルAIに字幕全体を要約させる
- 冒頭のあいさつ・宣伝・定型文を除外
- 概要、重要点3件、結論をダッシュボードのsummary欄に表示
- 外部の有料AI APIは使用しない
- 一度AI要約した動画は再利用する
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import requests
from youtube_transcript_api import YouTubeTranscriptApi


ROOT = Path(__file__).resolve().parents[1]
CHANNELS_PATH = ROOT / "channels.json"
OUTPUT_PATH = ROOT / "data" / "videos.json"
LOCAL_CONFIG_PATH = ROOT / "local_ai_config.json"

MAX_VIDEOS_PER_CHANNEL = int(os.getenv("MAX_VIDEOS_PER_CHANNEL", "8"))
TRANSCRIPT_WAIT_MIN = float(os.getenv("TRANSCRIPT_WAIT_MIN", "1.5"))
TRANSCRIPT_WAIT_MAX = float(os.getenv("TRANSCRIPT_WAIT_MAX", "3.0"))

DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_MAX_AI_SUMMARIES_PER_RUN = 4
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
RSS_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
SUMMARY_VERSION = 2

YTT_API = YouTubeTranscriptApi()

INTEREST_KEYWORDS = {
    "AI": ("ai", "生成ai", "chatgpt", "claude", "copilot", "エージェント"),
    "資産管理": ("資産", "投資", "株", "金利", "為替", "相続", "金融"),
    "日本経済": ("日本経済", "財政", "物価", "賃金", "産業", "景気"),
    "国際情勢": ("国際", "中国", "米国", "韓国", "欧州", "戦争", "移民"),
    "行政DX": ("自治体", "行政", "bpr", "dx", "デジタル", "業務改革"),
}


def clean_text(value: str | None) -> str:
    if not value:
        return ""

    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[[^\]]{0,30}\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_local_config() -> dict[str, Any]:
    defaults = {
        "model": DEFAULT_OLLAMA_MODEL,
        "maxSummariesPerRun": DEFAULT_MAX_AI_SUMMARIES_PER_RUN,
        "maxTranscriptChars": 30000,
    }

    if not LOCAL_CONFIG_PATH.exists():
        return defaults

    try:
        supplied = json.loads(
            LOCAL_CONFIG_PATH.read_text(encoding="utf-8")
        )
        defaults.update(supplied)
    except Exception as exc:
        print(
            f"local_ai_config.jsonの読込に失敗しました: {exc}",
            file=sys.stderr,
        )

    return defaults


def load_existing() -> dict[str, dict[str, Any]]:
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
        print(
            f"既存データの読み込みに失敗しました: {exc}",
            file=sys.stderr,
        )
        return {}


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_transcript(
    video_id: str,
) -> tuple[list[dict[str, Any]], str]:
    errors: list[str] = []

    def convert(fetched: Any) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []

        for snippet in fetched:
            text = clean_text(snippet.text)
            if not text:
                continue

            output.append(
                {
                    "text": text,
                    "start": float(getattr(snippet, "start", 0.0)),
                    "duration": float(
                        getattr(snippet, "duration", 0.0)
                    ),
                }
            )

        return output

    try:
        fetched = YTT_API.fetch(video_id, languages=["ja", "en"])
        snippets = convert(fetched)
        if snippets:
            return snippets, "ok"
    except Exception as exc:
        errors.append(str(exc))

    try:
        transcript_list = YTT_API.list(video_id)
        transcript = next(iter(transcript_list))
        fetched = transcript.fetch()
        snippets = convert(fetched)
        if snippets:
            return snippets, "ok"
    except Exception as exc:
        errors.append(str(exc))

    detail = " / ".join(errors)
    print(
        f"字幕取得失敗: {video_id}: "
        f"{detail[:800] or '字幕なし'}",
        file=sys.stderr,
    )
    return [], "unavailable"


def build_balanced_transcript(
    snippets: list[dict[str, Any]],
    max_chars: int,
    section_count: int = 10,
) -> str:
    """
    長い動画でも冒頭だけに偏らないよう、動画全体を時間帯別に分け、
    各区間から均等に字幕を採用します。
    """
    if not snippets:
        return ""

    lines = [
        f'[{format_timestamp(item["start"])}] {item["text"]}'
        for item in snippets
    ]
    full_text = "\n".join(lines)

    if len(full_text) <= max_chars:
        return full_text

    end_time = max(
        item["start"] + item.get("duration", 0.0)
        for item in snippets
    )
    end_time = max(end_time, 1.0)

    sections: list[list[str]] = [
        [] for _ in range(section_count)
    ]

    for item in snippets:
        ratio = item["start"] / end_time
        index = min(
            section_count - 1,
            max(0, int(ratio * section_count)),
        )
        sections[index].append(
            f'[{format_timestamp(item["start"])}] {item["text"]}'
        )

    quota = max(500, max_chars // section_count)
    selected: list[str] = []

    for index, section in enumerate(sections):
        if not section:
            continue

        # 冒頭区間はあいさつや宣伝が多いため、採用量を少し減らす。
        section_quota = int(quota * 0.65) if index == 0 else quota
        used = 0

        for line in section:
            if used + len(line) > section_quota:
                remaining = section_quota - used
                if remaining >= 80:
                    selected.append(line[:remaining])
                break

            selected.append(line)
            used += len(line) + 1

    return "\n".join(selected)[:max_chars]


def ollama_is_available() -> bool:
    try:
        response = requests.get(OLLAMA_TAGS_URL, timeout=5)
        return response.ok
    except requests.RequestException:
        return False


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.S,
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_local_ai(
    *,
    model: str,
    title: str,
    channel: str,
    transcript: str,
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "properties": {
            "overview": {"type": "string"},
            "keyPoints": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 3,
            },
            "conclusion": {"type": "string"},
        },
        "required": ["overview", "keyPoints", "conclusion"],
    }

    prompt = f"""
あなたはYouTube動画の編集者兼リサーチャーです。
次の字幕を、動画を見ていない人が重要部分を把握できるように
日本語で要約してください。

【最重要ルール】
- 字幕全体を見て、動画の中心的な主張を特定する
- 冒頭のあいさつ、自己紹介、チャンネル説明、宣伝、定型文は除外する
- 単に字幕の冒頭を短くしない
- タイトルの言い換えだけにしない
- 主張を支える理由、数字、具体例、変化、リスクを優先する
- 動画後半の結論や提言も確認する
- 字幕に存在しない情報を追加しない
- 話者の推測や意見は、事実と断定せず「話者は～とみている」と表す
- 重複する内容を3つの重要点に選ばない

【出力内容】
overview:
動画全体の要旨を100～180文字で記載する。

keyPoints:
互いに異なる重要点を3件記載する。
各項目は40～100文字とし、理由・数字・影響のいずれかを含める。

conclusion:
動画の結論、提言、または今後の見通しを50～120文字で記載する。

【動画】
チャンネル: {channel}
タイトル: {title}

【字幕】
{transcript}
""".strip()

    payload = {
        "model": model,
        "prompt": prompt,
        "format": schema,
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0.15,
            "num_ctx": 32768,
            "num_predict": 700,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=900,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollamaへの接続に失敗しました: {exc}"
        ) from exc

    # 古いOllamaでthinkまたはJSON Schemaが未対応の場合の再試行
    if response.status_code >= 400:
        fallback_payload = {
            "model": model,
            "prompt": (
                prompt
                + '\n\nJSONだけを返してください。'
                + '{"overview":"...",'
                + '"keyPoints":["...","...","..."],'
                + '"conclusion":"..."}'
            ),
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.15,
                "num_ctx": 32768,
                "num_predict": 700,
            },
        }
        response = requests.post(
            OLLAMA_URL,
            json=fallback_payload,
            timeout=900,
        )

    response.raise_for_status()
    response_text = str(response.json().get("response", ""))
    data = parse_json_response(response_text)

    overview = clean_text(str(data.get("overview", "")))
    conclusion = clean_text(str(data.get("conclusion", "")))
    key_points_raw = data.get("keyPoints", [])

    if not isinstance(key_points_raw, list):
        key_points_raw = []

    key_points = [
        clean_text(str(item))
        for item in key_points_raw
        if clean_text(str(item))
    ][:3]

    if not overview or len(key_points) < 3 or not conclusion:
        raise ValueError(
            "ローカルAIの要約に必要な項目が不足しています。"
        )

    return {
        "overview": overview,
        "keyPoints": key_points,
        "conclusion": conclusion,
    }


def format_dashboard_summary(ai: dict[str, Any]) -> str:
    points = ai["keyPoints"]
    return (
        f'【概要】{ai["overview"]} '
        f'【重要点】①{points[0]} ②{points[1]} ③{points[2]} '
        f'【結論】{ai["conclusion"]}'
    )


def balanced_extractive_fallback(
    snippets: list[dict[str, Any]],
    description: str,
) -> str:
    """
    Ollamaが使えない場合の無料フォールバック。
    冒頭だけでなく、動画の序盤・中盤・終盤から各1か所を選びます。
    """
    if not snippets:
        description = clean_text(description)
        if description:
            return (
                description[:360]
                + ("…" if len(description) > 360 else "")
            )
        return "字幕と概要欄を取得できませんでした。"

    end_time = max(
        item["start"] + item.get("duration", 0.0)
        for item in snippets
    )
    end_time = max(end_time, 1.0)

    buckets: list[list[dict[str, Any]]] = [[], [], []]

    for item in snippets:
        ratio = item["start"] / end_time
        index = min(2, int(ratio * 3))
        buckets[index].append(item)

    preferred = re.compile(
        r"結論|重要|ポイント|理由|結果|影響|今後|"
        r"つまり|一方|しかし|必要|増加|減少|\d"
    )
    chosen: list[str] = []

    for bucket_index, bucket in enumerate(buckets):
        candidates: list[tuple[int, str]] = []

        for item in bucket:
            text = clean_text(item["text"])
            if len(text) < 30:
                continue

            score = len(preferred.findall(text)) * 3 + min(
                len(text),
                160,
            ) // 40

            # 最初の区間は、開始60秒未満の字幕を低く評価する。
            if bucket_index == 0 and item["start"] < 60:
                score -= 4

            candidates.append((score, text))

        if candidates:
            candidates.sort(reverse=True)
            chosen.append(candidates[0][1][:150])

    if not chosen:
        chosen = [
            clean_text(item["text"])[:150]
            for item in snippets
            if len(clean_text(item["text"])) >= 30
        ][:3]

    return (
        "【重要箇所（AI未使用）】"
        + " ".join(
            f"{index + 1}.{text}"
            for index, text in enumerate(chosen)
        )
    )[:500]


def classify(
    title: str,
    summary: str,
    category: str,
) -> tuple[str, list[str]]:
    text = f"{title} {summary} {category}".lower()
    tags: list[str] = []

    for label, keywords in INTEREST_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            tags.append(label)

    priority = "high" if len(tags) >= 2 else "medium"

    if not tags:
        tags = ["新着動画"]

    return priority, tags[:4]


def get_thumbnail(entry: Any, video_id: str) -> str:
    media_group = entry.get("media_group") or []

    if media_group:
        thumbnails = media_group[0].get("media_thumbnail") or []
        if thumbnails and thumbnails[0].get("url"):
            return thumbnails[0]["url"]

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def is_current_ai_summary(
    existing: dict[str, Any] | None,
    model: str,
) -> bool:
    return bool(
        existing
        and existing.get("summary")
        and existing.get("summarySource") == "local_ai_ollama"
        and int(existing.get("summaryVersion", 0)) == SUMMARY_VERSION
        and existing.get("summaryModel") == model
    )


def parse_entry(
    channel: dict[str, Any],
    entry: Any,
) -> dict[str, Any] | None:
    video_id = entry.get("yt_videoid") or entry.get("videoid")

    if not video_id:
        match = re.search(
            r"(?:v=|/)([\w-]{11})(?:[?&/]|$)",
            entry.get("link", ""),
        )
        video_id = match.group(1) if match else None

    if not video_id:
        return None

    description = ""
    media_group = entry.get("media_group") or []

    if media_group:
        description = media_group[0].get("media_description", "")

    return {
        "videoId": video_id,
        "channelData": channel,
        "entry": entry,
        "title": clean_text(entry.get("title")),
        "description": clean_text(
            description or entry.get("summary", "")
        ),
        "published": entry.get("published") or entry.get("updated"),
    }


def update() -> None:
    config = load_local_config()
    model = str(config.get("model", DEFAULT_OLLAMA_MODEL))
    max_ai_per_run = max(
        1,
        int(
            config.get(
                "maxSummariesPerRun",
                DEFAULT_MAX_AI_SUMMARIES_PER_RUN,
            )
        ),
    )
    max_transcript_chars = max(
        8000,
        int(config.get("maxTranscriptChars", 30000)),
    )

    channels = json.loads(
        CHANNELS_PATH.read_text(encoding="utf-8")
    )
    existing_map = load_existing()

    candidates: list[dict[str, Any]] = []
    errors: list[str] = []

    for channel in channels:
        feed_url = RSS_TEMPLATE.format(
            channel_id=channel["channelId"]
        )
        feed = feedparser.parse(feed_url)

        if getattr(feed, "bozo", False) and not feed.entries:
            errors.append(f'{channel["name"]}: RSS取得失敗')
            continue

        for entry in feed.entries[:MAX_VIDEOS_PER_CHANNEL]:
            candidate = parse_entry(channel, entry)
            if candidate:
                candidates.append(candidate)

    candidates.sort(
        key=lambda item: item.get("published") or "",
        reverse=True,
    )

    ollama_available = ollama_is_available()

    if ollama_available:
        print(f"ローカルAIを使用します: {model}")
    else:
        print(
            "Ollamaに接続できないため、抽出型要約を使用します。",
            file=sys.stderr,
        )

    videos: list[dict[str, Any]] = []
    ai_generated_count = 0
    reused_count = 0
    transcript_fetch_count = 0

    for candidate in candidates:
        video_id = candidate["videoId"]
        channel = candidate["channelData"]
        entry = candidate["entry"]
        title = candidate["title"]
        description = candidate["description"]
        existing = existing_map.get(video_id)

        if is_current_ai_summary(existing, model):
            summary = existing["summary"]
            key_points = existing.get("keyPoints", [])
            conclusion = existing.get("conclusion", "")
            summary_source = "local_ai_ollama"
            transcript_status = existing.get(
                "transcriptStatus",
                "ok",
            )
            priority = existing.get("priority", "medium")
            tags = existing.get("tags", ["新着動画"])
            reused_count += 1
        elif (
            ollama_available
            and ai_generated_count >= max_ai_per_run
            and existing
            and existing.get("summary")
        ):
            # 1回の実行時間を抑えるため、残りは次回以降にAI化する。
            summary = existing["summary"]
            key_points = existing.get("keyPoints", [])
            conclusion = existing.get("conclusion", "")
            summary_source = existing.get(
                "summarySource",
                "pending_local_ai",
            )
            transcript_status = existing.get(
                "transcriptStatus",
                "ok",
            )
            priority = existing.get("priority", "medium")
            tags = existing.get("tags", ["新着動画"])
        else:
            if transcript_fetch_count:
                time.sleep(
                    random.uniform(
                        TRANSCRIPT_WAIT_MIN,
                        TRANSCRIPT_WAIT_MAX,
                    )
                )

            snippets, transcript_status = fetch_transcript(video_id)
            transcript_fetch_count += 1
            key_points: list[str] = []
            conclusion = ""

            if snippets and ollama_available:
                transcript = build_balanced_transcript(
                    snippets,
                    max_chars=max_transcript_chars,
                )

                try:
                    ai = call_local_ai(
                        model=model,
                        title=title,
                        channel=channel["name"],
                        transcript=transcript,
                    )
                    summary = format_dashboard_summary(ai)
                    key_points = ai["keyPoints"]
                    conclusion = ai["conclusion"]
                    summary_source = "local_ai_ollama"
                    ai_generated_count += 1
                except Exception as exc:
                    print(
                        f"ローカルAI要約失敗: {video_id}: {exc}",
                        file=sys.stderr,
                    )
                    summary = balanced_extractive_fallback(
                        snippets,
                        description,
                    )
                    summary_source = "transcript_balanced_fallback"
            else:
                summary = balanced_extractive_fallback(
                    snippets,
                    description,
                )
                summary_source = (
                    "transcript_balanced_fallback"
                    if snippets
                    else "description_fallback"
                )

            priority, tags = classify(
                title,
                summary,
                channel.get("category", ""),
            )

        videos.append(
            {
                "videoId": video_id,
                "channel": channel["name"],
                "channelHandle": channel.get("handle", ""),
                "title": title,
                "published": candidate["published"],
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "thumbnail": get_thumbnail(entry, video_id),
                "description": description,
                "summary": summary,
                "keyPoints": key_points,
                "conclusion": conclusion,
                "priority": priority,
                "tags": tags,
                "summarySource": summary_source,
                "summaryVersion": (
                    SUMMARY_VERSION
                    if summary_source == "local_ai_ollama"
                    else 0
                ),
                "summaryModel": (
                    model
                    if summary_source == "local_ai_ollama"
                    else ""
                ),
                "transcriptStatus": transcript_status,
                "durationMinutes": None,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "channelCount": len(channels),
                "videos": videos,
                "errors": errors,
                "message": (
                    "本日の新着なし"
                    if not videos and not errors
                    else ""
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{len(videos)}件を保存しました。"
        f" AI要約={ai_generated_count}件、"
        f"AI要約再利用={reused_count}件、"
        f"字幕取得={transcript_fetch_count}件"
    )

    if (
        ollama_available
        and ai_generated_count >= max_ai_per_run
    ):
        print(
            "未変換の過去動画は次回以降、"
            "新しい動画を優先しながら順次AI要約します。"
        )

    if errors:
        print("エラー: " + " / ".join(errors), file=sys.stderr)


if __name__ == "__main__":
    update()
