#!/usr/bin/env python3
"""
YouTube RSSから登録チャンネルの新着動画を取得し、
公開字幕・自動生成字幕から「抽出型要約」を無料で作成します。

外部の生成AI APIやAPIキーは使用しません。
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
from youtube_transcript_api import YouTubeTranscriptApi


ROOT = Path(__file__).resolve().parents[1]
CHANNELS_PATH = ROOT / "channels.json"
OUTPUT_PATH = ROOT / "data" / "videos.json"

MAX_VIDEOS_PER_CHANNEL = int(os.getenv("MAX_VIDEOS_PER_CHANNEL", "8"))
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "320"))
RSS_TEMPLATE = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"

YTT_API = YouTubeTranscriptApi()

STOP_PHRASES = {
    "こんにちは",
    "こんばんは",
    "おはようございます",
    "ありがとうございます",
    "よろしくお願いします",
    "チャンネル登録",
    "高評価",
    "コメント欄",
    "概要欄",
    "ご視聴",
}

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
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_for_compare(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[\s、。！？!?,.・「」『』【】（）()：:ー\-]", "", text)
    return text


def char_ngrams(text: str, size: int = 2) -> list[str]:
    normalized = normalize_for_compare(text)
    return [
        normalized[index:index + size]
        for index in range(max(0, len(normalized) - size + 1))
        if len(normalized[index:index + size]) == size
    ]


def split_into_sentences(text: str) -> list[str]:
    text = clean_text(text)
    if not text:
        return []

    # 字幕は句読点が少ないことがあるため、通常の文分割後に長文も分割する
    rough = re.split(r"(?<=[。！？!?])\s+|[\r\n]+", text)
    sentences: list[str] = []

    for part in rough:
        part = clean_text(part)
        if not part:
            continue

        if len(part) <= 180:
            sentences.append(part)
            continue

        # 句読点がない長い字幕を読点や接続部分で切る
        chunks = re.split(
            r"(?<=[、,])|(?=しかし|一方で|つまり|そのため|そして|では|結論として)",
            part,
        )
        current = ""

        for chunk in chunks:
            chunk = clean_text(chunk)
            if not chunk:
                continue

            if current and len(current) + len(chunk) > 150:
                sentences.append(current)
                current = chunk
            else:
                current += chunk

        if current:
            sentences.append(current)

    # 極端に短い断片は前後と結合
    merged: list[str] = []
    for sentence in sentences:
        if merged and len(sentence) < 30:
            merged[-1] = clean_text(merged[-1] + " " + sentence)
        else:
            merged.append(sentence)

    return merged


def fetch_transcript(video_id: str) -> tuple[str, str]:
    errors: list[str] = []

    try:
        fetched = YTT_API.fetch(video_id, languages=["ja", "en"])
        text = clean_text(" ".join(snippet.text for snippet in fetched))
        if text:
            return text, "ok"
    except Exception as exc:
        errors.append(str(exc))

    # 日本語・英語以外しかない場合も、取得可能な字幕を試す
    try:
        transcript_list = YTT_API.list(video_id)
        transcript = next(iter(transcript_list))
        fetched = transcript.fetch()
        text = clean_text(" ".join(snippet.text for snippet in fetched))
        if text:
            return text, "ok"
    except Exception as exc:
        errors.append(str(exc))

    detail = " / ".join(errors)
    print(
        f"字幕取得失敗: {video_id}: {detail[:500] or '字幕なし'}",
        file=sys.stderr,
    )
    return "", "unavailable"


def sentence_similarity(first: str, second: str) -> float:
    first_set = set(char_ngrams(first))
    second_set = set(char_ngrams(second))

    if not first_set or not second_set:
        return 0.0

    return len(first_set & second_set) / len(first_set | second_set)


def is_unhelpful(sentence: str) -> bool:
    normalized = normalize_for_compare(sentence)

    if len(normalized) < 22:
        return True

    phrase_hits = sum(
        1 for phrase in STOP_PHRASES
        if normalize_for_compare(phrase) in normalized
    )
    return phrase_hits >= 2


def extractive_summary(title: str, transcript: str) -> str:
    sentences = [
        sentence
        for sentence in split_into_sentences(transcript)
        if not is_unhelpful(sentence)
    ]

    if not sentences:
        return ""

    corpus_ngrams = Counter()
    for sentence in sentences:
        corpus_ngrams.update(char_ngrams(sentence))

    title_ngrams = set(char_ngrams(title))
    total = len(sentences)
    scored: list[tuple[float, int, str]] = []

    for index, sentence in enumerate(sentences):
        grams = char_ngrams(sentence)
        unique_grams = set(grams)

        if not grams:
            continue

        # 文中の頻出語を評価。ただし極端に多い一般語は影響を抑える
        frequency_score = sum(
            min(corpus_ngrams[gram], 8)
            for gram in unique_grams
        ) / max(1, len(unique_grams))

        title_score = (
            len(unique_grams & title_ngrams)
            / max(1, len(title_ngrams))
        )

        position = index / max(1, total - 1)
        position_score = 0.0
        if position <= 0.18:
            position_score += 0.45
        if position >= 0.80:
            position_score += 0.55

        number_score = 0.35 if re.search(r"\d", sentence) else 0.0
        conclusion_score = 0.50 if re.search(
            r"結論|つまり|要するに|重要|ポイント|今後|必要|理由",
            sentence,
        ) else 0.0

        length_score = 1.0 - min(abs(len(sentence) - 100) / 130, 0.8)

        score = (
            frequency_score * 0.35
            + title_score * 2.8
            + position_score
            + number_score
            + conclusion_score
            + length_score
        )
        scored.append((score, index, sentence))

    scored.sort(reverse=True)

    selected: list[tuple[int, str]] = []
    current_chars = 0

    for _, index, sentence in scored:
        if any(
            sentence_similarity(sentence, chosen) >= 0.48
            for _, chosen in selected
        ):
            continue

        remaining = SUMMARY_MAX_CHARS - current_chars
        if remaining < 45:
            break

        clipped = sentence
        if len(clipped) > remaining:
            clipped = clipped[:remaining].rstrip("、。 ") + "…"

        selected.append((index, clipped))
        current_chars += len(clipped)

        if len(selected) >= 3:
            break

    if not selected:
        return ""

    # 動画中の順番に並べ直し、読みやすく表示
    selected.sort(key=lambda item: item[0])

    parts = []
    for _, sentence in selected:
        sentence = sentence.strip()
        if sentence and sentence[-1] not in "。！？!?…":
            sentence += "。"
        parts.append(sentence)

    return " ".join(parts)


def fallback_summary(title: str, description: str) -> str:
    description = clean_text(description)

    if description:
        sentences = split_into_sentences(description)
        usable = [sentence for sentence in sentences if not is_unhelpful(sentence)]
        source = " ".join(usable[:3]) if usable else description
        return source[:SUMMARY_MAX_CHARS] + (
            "…" if len(source) > SUMMARY_MAX_CHARS else ""
        )

    return (
        f"「{title}」に関する新着動画です。"
        "字幕と概要欄から内容を取得できなかったため、詳細はリンク先で確認してください。"
    )


def classify(title: str, summary: str, category: str) -> tuple[str, list[str]]:
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


def update() -> None:
    channels = json.loads(CHANNELS_PATH.read_text(encoding="utf-8"))
    videos: list[dict[str, Any]] = []
    errors: list[str] = []

    for channel in channels:
        feed_url = RSS_TEMPLATE.format(channel_id=channel["channelId"])
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
            description = clean_text(description or entry.get("summary", ""))

            transcript, transcript_status = fetch_transcript(video_id)

            if transcript:
                summary = extractive_summary(title, transcript)
                summary_source = "transcript_extractive"
            else:
                summary = fallback_summary(title, description)
                summary_source = "description_fallback"

            if not summary:
                summary = fallback_summary(title, description)
                summary_source = "description_fallback"

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
                    "published": entry.get("published") or entry.get("updated"),
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail": get_thumbnail(entry, video_id),
                    "description": description,
                    "summary": summary,
                    "priority": priority,
                    "tags": tags,
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
