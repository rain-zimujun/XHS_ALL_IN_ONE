from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlsplit

from backend.app.adapters.xhs.request_env import direct_xhs_request_env

_TOPIC_ID_RE = re.compile(r"[0-9a-f]{24}")
_TOPIC_PATH_RE = re.compile(r"/topics?/([0-9a-f]{24})")
_TOPIC_QUERY_RE = re.compile(r"(?:topic_id|topicId)=([0-9a-f]{24})")


def parse_topic_ref(text: str) -> tuple[str, str]:
    """Split a topic reference into ``(topic_id, name)``.

    Accepts a topic page link, a bare topic id, or a plain name. The creator
    topic-search endpoint currently rejects our signature (HTTP 406), so a link
    or id lets the caller bypass the lookup entirely.
    """
    value = (text or "").strip()
    if not value:
        return "", ""
    if "://" in value or value.startswith("/"):
        match = _TOPIC_PATH_RE.search(value) or _TOPIC_QUERY_RE.search(value)
        if match:
            name = ""
            try:
                query = parse_qs(urlsplit(value).query)
                name = (query.get("name") or query.get("topic_name") or [""])[0]
            except ValueError:
                name = ""
            return match.group(1), name.strip()
        return "", value
    if _TOPIC_ID_RE.fullmatch(value):
        return value, ""
    return "", value


def resolve_topics(api, topics: list[str] | None) -> tuple[list[dict], list[str], list[str]]:
    """Map topic references to platform topic entities, degrading to plain text.

    References that already carry an id (pasted topic link or bare id) become
    entities directly. Names still go through the lookup endpoint; when it
    fails, the text is posted as a plain hashtag instead of aborting the publish.
    """
    hash_tags: list[dict] = []
    plain: list[str] = []
    warnings: list[str] = []
    for topic in topics or []:
        if isinstance(topic, dict) and topic.get("id"):
            # Already resolved (topic cache or pasted link): use it as-is.
            hash_tags.append({
                "id": str(topic["id"]),
                "link": str(topic.get("link") or ""),
                "name": str(topic.get("name") or "话题"),
                "type": "topic",
            })
            continue
        if not isinstance(topic, str) or not topic.strip():
            continue
        topic_id, ref_name = parse_topic_ref(topic)
        if topic_id:
            hash_tags.append({
                "id": topic_id,
                "link": "",
                "name": ref_name or "话题",
                "type": "topic",
            })
            if not ref_name:
                warnings.append(f"话题 {topic_id} 没有提供名称，已用默认名称关联")
            continue
        name = topic.strip()
        try:
            success, _message, payload = api.get_topic(name)
        except Exception:  # a broken lookup must not abort the publish
            success, payload = False, None
        items = []
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                items = data.get("topic_info_dtos") or []
        if not success or not items:
            plain.append(name)
            warnings.append(f"话题「{name}」未能关联到话题实体，已按普通文字写入正文")
            continue
        item = items[0]
        if not isinstance(item, dict) or not item.get("id"):
            plain.append(name)
            warnings.append(f"话题「{name}」数据不完整，已按普通文字写入正文")
            continue
        hash_tags.append({
            "id": item["id"],
            "link": item.get("link", ""),
            "name": item.get("name") or name,
            "type": "topic",
        })
    return hash_tags, plain, warnings


class XhsCreatorApiAdapter:
    def __init__(self, cookies: str) -> None:
        self.cookies = cookies

    def _api(self):
        from apis.xhs_creator_apis import XHS_Creator_Apis
        from xhs_utils.xhs_creator import XHSCreatorAuth

        auth = XHSCreatorAuth.from_cookie(self.cookies)
        return XHS_Creator_Apis(auth)

    def get_topic(self, keyword: str) -> Any:
        with direct_xhs_request_env():
            return self._api().get_topic(keyword=keyword)

    def get_location_info(self, keyword: str) -> Any:
        with direct_xhs_request_env():
            return self._api().get_location_info(keyword=keyword)

    def get_published_notes(self) -> Any:
        with direct_xhs_request_env():
            return self._api().get_all_posted_notes()

    def upload_media(self, file_path: str, media_type: str) -> dict[str, Any]:
        file_data = self._resolve_file_data(file_path)
        with direct_xhs_request_env():
            success, message, payload = self._api().upload_media(file_data, media_type)
        if not success:
            raise RuntimeError(message or "Creator media upload failed")
        return payload or {}

    @staticmethod
    def _resolve_file_data(file_path: str) -> bytes:
        import pathlib
        raw_bytes: bytes | None = None

        if file_path.startswith("http://") or file_path.startswith("https://"):
            import requests
            resp = requests.get(file_path, timeout=30, headers={"Referer": ""})
            resp.raise_for_status()
            raw_bytes = resp.content
        elif file_path.startswith("/api/files/media/"):
            from backend.app.core.config import get_settings
            file_name = file_path.split("/")[-1]
            local = pathlib.Path(get_settings().storage_dir) / "media" / file_name
            if local.is_file():
                raw_bytes = local.read_bytes()
        else:
            p = pathlib.Path(file_path)
            if p.is_file():
                raw_bytes = p.read_bytes()

        if raw_bytes is None:
            raise FileNotFoundError(f"素材文件不存在: {file_path}")

        lower = file_path.lower()
        if lower.endswith(".webp") or (len(raw_bytes) > 4 and raw_bytes[:4] == b"RIFF"):
            try:
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=92)
                raw_bytes = buf.getvalue()
            except Exception:
                pass

        return raw_bytes

    def post_note(self, note_info: dict[str, Any]) -> dict[str, Any]:
        with direct_xhs_request_env():
            if note_info.get("media_type") == "image" and note_info.get("image_file_infos"):
                return self._post_uploaded_image_note(note_info)

            success, message, payload = self._api().post_note(note_info)
        if not success:
            raise RuntimeError(f"{message or 'Creator note publish failed'} | raw={str(payload)[:300]}")
        return payload or {}

    def _post_uploaded_image_note(self, note_info: dict[str, Any]) -> dict[str, Any]:
        from apis.xhs_creator_apis import XHS_Creator_Apis
        from xhs_utils.http_util import REQUEST_TIMEOUT
        from xhs_utils.xhs_creator import XHSCreatorAuth
        from xhs_utils.xhs_creator_util import (
            get_post_note_image_data,
            load_creator_rap_fingerprint_hex,
        )
        from xhs_utils.xhs_pc.params import generate_x_rap_param

        auth = XHSCreatorAuth.from_cookie(self.cookies)
        api = XHS_Creator_Apis(auth)
        post_api = "/web_api/sns/v2/note"
        post_loc = {}
        location = note_info.get("location")
        if isinstance(location, dict):
            post_loc = location
        elif isinstance(location, str) and location.strip():
            success, message, location_info = api.get_location_info(location.strip())
            if not success:
                raise RuntimeError(message or "Creator location lookup failed")
            poi_list = (location_info.get("data") or {}).get("poi_list") or []
            if not poi_list:
                raise RuntimeError("未找到该地点")
            poi = poi_list[0]
            post_loc = {
                "name": poi["name"],
                "subname": poi["full_address"],
                "poi_id": poi["poi_id"],
                "poi_type": poi["poi_type"],
            }
        data = get_post_note_image_data(
            note_info.get("title", ""),
            note_info.get("desc", ""),
            note_info.get("postTime"),
            post_loc,
            note_info.get("type", 1),
            note_info["image_file_infos"],
        )
        # 浏览器实抓：未选地点时省略 post_loc 键，空对象会被服务端以参数错误打回
        if not post_loc:
            data["common"].pop("post_loc", None)
        hash_tags, plain_topics, topic_warnings = resolve_topics(api, note_info.get("topics"))
        for insert_topic in hash_tags:
            data["common"]["hash_tag"].append(insert_topic)
            data["common"]["desc"] += f" #{insert_topic['name']}[话题]# "
        for name in plain_topics:
            # Plain text fallback: the note still carries the hashtag text even
            # when the platform cannot resolve it into a topic entity.
            data["common"]["desc"] += f" #{name} "
        headers, cookies, body = api._request_params(
            post_api,
            data,
            "POST",
            referer=f"{api.base_url}/",
            target_origin=api.edith_url,
            order_wire_headers=False,
        )
        headers["x-rap-param"] = generate_x_rap_param(
            post_api,
            body,
            fingerprint_hex=load_creator_rap_fingerprint_hex(),
        )
        response = api.http.post(
            api.edith_url + post_api,
            headers=headers,
            data=body.encode("utf-8"),
            cookies=cookies,
            timeout=REQUEST_TIMEOUT,
        )
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(
                payload.get("msg") or payload.get("message") or f"Creator note publish failed: {str(payload)[:300]}"
            )
        if topic_warnings and isinstance(payload, dict):
            payload["_topic_warnings"] = topic_warnings
        return payload
