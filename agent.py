from __future__ import annotations

import json
import re
import threading
from difflib import SequenceMatcher
from os import getenv
from time import monotonic, sleep
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

from src.agent.proxy_client import ProxyClient

CALLS_PER_TURN, OPEN_BATCH, TURN_RESERVE, EARLY_TURNS = 16, 5, 6, 10
OPEN_CALLS = CALLS_PER_TURN
OPEN_LIMIT, SEARCH_LIMIT, JUDGE_CARDS, JUDGE_LIMIT, SEARCH_ROUNDS, BEST_ROUNDS, FIGURE_LIMIT = 400, 160, 18, 120, 8, 6, 150
OPEN_SPARE = 40
EPISODE_SECONDS, ORDER_NOW, MODEL_STOP, TALK_END, STORE_STOP, SEND_END = 180.0, 230.0, 225.0, 262.0, 250.0, 285.0
SEND_PAUSES, BUSY = (4.0, 8.0, 16.0), (None, 429, 500, 502, 503, 504)
WRITER_SECONDS, STALE_SHARE, STALE_TURNS, PLAN_TURNS = 25.0, 0.25, 2, 2
OVER_REACH, OPEN_REACH, QUERY_LENGTH = 1.35, 1.6, 200
ASK_BLIND, ASK_NARROW, ASK_LIMIT, ASK_SHELF, ASK_ROW = 10, 3, 11, 30, 12
FIT_BASE, COLOUR_PROOF = 3, 8
FIRM_FIT, SURE_FIT, LIKELY_FIT, UNSURE_FIT, FAINT_FIT = 1.0, 0.85, 0.5, 0.3, 0.1
LISTED_FIT, LISTED_SHORT, LISTED_LONG = 0.7, 0.6, 0.5
DECIDE_CARDS, DECIDE_LIMIT, DECIDE_STOP, DECIDE_SPARE = 9, 6, 220.0, 2
STORE_ENTRIES, STORE_LIMIT, SWEEP_ROUNDS, SWEEP_MAKERS = 40, 10, 3, 3
NO_MAKER = ("none", "no brand", "oem")
SAID_NEEDED = r"(?:(?:a|an|is|be|it|the)\s+)*(?:required|requirement|needed|necessary|essential|must|need|included|present|chosen|selected|provided)"
NOT_A_PROPERTY = re.compile(r"categor|budget|price|\bstock|availab")
OF_THE_SALE = re.compile(r"warrant|guarantee|^condition$|^model\b|\bseries$|seller|shipping")
TAKES_BACK = re.compile(r"\b(?:instead|chang\w+|switch\w*|swap\w*|forget|no longer|rather than|replac\w+)\b")
PLAIN_QUESTION = "What will you use it for?"
USE_WORDS = "what you will use it for"
ASK_ALONE = "What {} do you need?"

Key = Tuple[str, str]


class Engine:
    def __init__(self, router: str, router_settings: Dict[str, Any], chutes: str = "", chutes_settings: Optional[Dict[str, Any]] = None) -> None:
        self.router, self.router_settings = router, dict(router_settings)
        self.chutes, self.chutes_settings = chutes, dict(chutes_settings or {})
    def route(self, on_chutes: bool) -> Tuple[str, Dict[str, Any]]:
        return (self.chutes, self.chutes_settings) if on_chutes else (self.router, self.router_settings)
    @property
    def thinks(self) -> bool:
        return (self.router_settings.get("reasoning") or {}).get("enabled", True) is not False and "reasoning" in self.router_settings


GLM = Engine("z-ai/glm-5.2", {"temperature": 0}, "zai-org/GLM-5.2-TEE", {"temperature": 0})
GLM_WARM = Engine("z-ai/glm-5.2", {"temperature": 0.7}, "zai-org/GLM-5.2-TEE", {"temperature": 0.7})
GLM_MILD = Engine("z-ai/glm-5.2", {"temperature": 0.3}, "zai-org/GLM-5.2-TEE", {"temperature": 0.3})
LUNA = Engine("openai/gpt-6-luna", {})
GLM_FLASH = Engine("z-ai/glm-5.3-flash", {"temperature": 0, "reasoning": {"effort": "medium"}})
GLM_DEEP = Engine("z-ai/glm-5.3-flash", {"temperature": 0, "reasoning": {"effort": "high"}})
QWEN = Engine("qwen/qwen3.8-27b", {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
                                   "tool_choice": "required"})
WRITERS = (GLM, GLM_WARM, QWEN)
FIRST_WRITERS = (GLM, GLM_MILD)
VOTERS, VOTE_STOP = (GLM, GLM), 170.0
THINK_STOP = 205.0


class Text:
    memo: Dict[Any, Any] = {}
    @staticmethod
    def shows_amount(words: Any, unit: Any = "") -> bool:
        said = " ".join(str(words or "").lower().replace("\\", "").split())
        if "=" in said:
            label, _, value = said.partition("=")
            ends = re.findall(r"[a-z]+", label)
            named = " ".join(str(unit or "").lower().split()).rstrip("s")
            if len(ends) >= 2 and (len(ends[-1]) <= 3 or (named and ends[-1].rstrip("s") == named)):
                return True
            return Text.shows_amount(value, unit)
        if re.search(r"\d[a-z]", said) or re.search(r"\d\s?(?:''|\"|”|″)", said):
            return True
        named = " ".join(str(unit or "").lower().split()).rstrip("s")
        for lead, after in re.findall(r"(\S?)\d+(?:[.,]\d+)?\s([a-z]+)", said):
            stuck = bool(lead) and (lead.isalpha() or lead == "-")
            if not stuck and (len(after) <= 4 or (named and (after.rstrip("s") == named or after.startswith(named) or named.startswith(after)))):
                return True
        return False
    @staticmethod
    def unit_elsewhere(words: Any, texts: Any, unit: Any = "") -> bool:
        named = " ".join(str(unit or "").lower().split()).rstrip("s")
        page = " ".join(Text.lowered(text) for text in texts)
        for number in set(re.findall(r"\d+(?:[.,]\d+)?", Text.lowered(words))):
            for after in re.findall(r"(?<![\d.,])" + re.escape(number) + r"(?![\d])(\s?)([a-z]*)", page):
                if after[1] and (not after[0] or (named and (after[1].rstrip("s") == named or after[1].startswith(named)))):
                    return True
        return False
    @staticmethod
    def lowered(text: Any) -> str:
        raw = text if isinstance(text, str) else str(text or "")
        done = Text.memo.get(("lowered", raw))
        if done is None:
            plain = re.sub(r"(?<=\d)\s?(?:''|″|”|\")(?=[\s,;:/)\]x×*-]|$)", '"', raw.lower())
            plain = re.sub(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d]|,\d)", lambda m: m.group(0).replace(",", ""), plain)
            done = Text.memo[("lowered", raw)] = " ".join(plain.split())
        return done
    @staticmethod
    def middle(figures: List[float]) -> float:
        ordered = sorted(figures)
        if not ordered:
            return 0.0
        half = len(ordered) // 2
        return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2
    @staticmethod
    def word_list(text: Any) -> List[str]:
        raw = text if isinstance(text, str) else str(text or "")
        done = Text.memo.get(("words", raw))
        if done is None:
            done = Text.memo[("words", raw)] = tuple(re.findall(r'\d+(?:\.\d+)?"|[a-z0-9]+(?:\.[0-9]+)?', Text.lowered(raw)))
        return list(done)
    @staticmethod
    def singular(word: str) -> str:
        done = Text.memo.get(("singular", word))
        if done is None:
            done = Text.memo[("singular", word)] = Text._singular(word)
        return done
    @staticmethod
    def _singular(word: str) -> str:
        if len(word) > 4 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
            return word[:-2]
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) > 4 and word.endswith("ie"):
            return word[:-2] + "y"
        if len(word) > 4 and word.endswith(("ve", "fe")):
            return word[:-2] + "f"
        return word
    @staticmethod
    def closed_up(text: Any) -> str:
        raw = text if isinstance(text, str) else str(text or "")
        done = Text.memo.get(("closed", raw))
        if done is None:
            done = Text.memo[("closed", raw)] = re.sub(r"(\d)\s+(?=[a-z]{1,4}\b)", r"\1", Text.lowered(raw))
        return done
    @staticmethod
    def same_word(one: str, other: str) -> bool:
        if one == other:
            return True
        short, long = sorted((one, other), key=len)
        if len(short) >= 4 and not short[0].isdigit() and long.startswith(short) and len(long) - len(short) <= 2:
            return True
        return (len(short) >= 7 and len(long) == len(short) + 1 and short.isalpha() and long[:2] == short[:2] and long[-1] == short[-1]
                and any(long[:i] + long[i + 1:] == short for i in range(2, len(long) - 1)))
    @staticmethod
    def stems(text: Any) -> Set[str]:
        raw = text if isinstance(text, str) else str(text or "")
        done = Text.memo.get(("stems", raw))
        if done is None:
            done = Text.memo[("stems", raw)] = frozenset(Text.singular(w) for w in Text.word_list(Text.closed_up(raw)))
        return set(done)
    @staticmethod
    def label_key(text: Any) -> str:
        raw = str(text)
        done = Text.memo.get(("label", raw))
        if done is None:
            done = Text.memo[("label", raw)] = " ".join(Text.singular(w) for w in Text.word_list(raw.replace("&", " and ")) if w != "and")
        return done
    @staticmethod
    def money(value: Any) -> Optional[float]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None
    @staticmethod
    def content_words(text: str) -> Set[str]:
        return {Text.singular(w) for w in Text.word_list(text) if len(w) >= 4} - {"that", "what", "with", "this", "from", "have", "them", "they", "their"}
    @staticmethod
    def plain_quotes(text: Any) -> str:
        return Text.lowered(text).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    @staticmethod
    def one_letter_apart(one: str, two: str) -> bool:
        if one == two or min(len(one), len(two)) < 4 or abs(len(one) - len(two)) > 1 or " " in one + two:
            return False
        if len(one) == len(two):
            return sum(a != b for a, b in zip(one, two)) == 1
        short, long_ = sorted((one, two), key=len)
        return any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))
    @staticmethod
    def shortened(form: str, value: str) -> bool:
        form, value = Text.lowered(form).strip(), Text.lowered(value).strip()
        if re.search(r"\d", form) or " " in form or len(form) > len(value) - 2:
            return False
        letters = iter(value)
        return all(c in letters for c in form)
    @staticmethod
    def word_set(query: str) -> FrozenSet[Any]:
        return frozenset(Text.singular(w) for w in Text.word_list(query))
    @staticmethod
    def name_words(text: Any) -> Set[str]:
        raw = str(text)
        done = Text.memo.get(("names", raw))
        if done is None:
            words = {Text.singular(w) for w in Text.word_list(raw.replace("_", " "))} - {"of"}
            done = Text.memo[("names", raw)] = frozenset("color" if w == "colour" else w for w in words)
        return set(done)
    @staticmethod
    def entries(text: Any) -> List[str]:
        raw = text if isinstance(text, str) else str(text or "")
        done = Text.memo.get(("entries", raw))
        if done is None:
            parts = (" ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(part))) for part in re.split(r"[,;/|]+", Text.lowered(raw)))
            done = Text.memo[("entries", raw)] = tuple(part for part in parts if part)
        return list(done)
    @staticmethod
    def other_words(tokens: Set[str]) -> Set[str]:
        return {w for w in tokens if re.fullmatch(r"[a-z]{3,}", w) and not re.fullmatch(r"x*[sml]|x+l", w)}
    @staticmethod
    def same_unit(one: str, other: str) -> bool:
        done = Text.memo.get(("same unit", one, other))
        if done is None:
            done = Text.memo[("same unit", one, other)] = Text._same_unit(one, other)
        return done
    @staticmethod
    def _same_unit(one: str, other: str) -> bool:
        unit = Text.unit_word
        a, b = sorted((unit(Text.lowered(one)), unit(Text.lowered(other))), key=len)
        if not a or a[0] != b[0] or len(b) <= 3 or (len(a) == 1 and len(b) > 4):
            return bool(a) and a == b
        rest = iter(b)
        return all(letter in rest for letter in a)
    @staticmethod
    def unit_word(word: str) -> str:
        done = Text.memo.get(("unit", word))
        if done is None:
            done = Text.memo[("unit", word)] = word[:-1] if len(word) == 3 and word.endswith("s") else Text.singular(word)
        return done
    @staticmethod
    def says_value(line: str, value: str) -> bool:
        figures = re.findall(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
        if figures:
            said = {float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(line).replace(",", ""))}
            return all(float(f) in said for f in figures)
        said_words = Text.stems(line) | set(Text.word_list(Text.lowered(line)))
        words = [w for w in Text.word_list(Text.lowered(value)) if len(w) >= 3]
        return not words or any(w in said_words or Text.stems(w) & said_words for w in words)
    @staticmethod
    def says_amount(text: str, amount: float) -> bool:
        figures = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]
        return any(abs(figure - amount) < 0.005 for figure in figures)
    @staticmethod
    def spoken(name: str) -> str:
        words = str(name).split()
        last = words[-1] if words else ""
        if last.endswith(("series", "species")):
            pass
        elif len(last) > 4 and last.endswith("ies"):
            last = last[:-3] + "y"
        elif len(last) > 4 and last.endswith(("ches", "shes", "sses", "xes")):
            last = last[:-2]
        elif len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is")):
            last = last[:-1]
        return " ".join(words[:-1] + [last])
    @staticmethod
    def label_name(label: Any, entries: Any = ()) -> str:
        words = Text.word_list(str(label).replace("_", " "))
        if len(words) >= 3 and words[:2] == ["type", "of"]:
            words = words[2:] + ["type"]
        units = {m.group(1) for entry in entries for m in re.finditer(r"\d(?:\.\d+)?\s?([a-z]+)\b", Text.lowered(entry))}
        while len(words) >= 2 and (len(words[-1]) <= 2 or words[-1] in units):
            words = words[:-1]
        return " ".join(Text.spoken(word) for word in words)
    @staticmethod
    def same_amount(value: str, form: Any) -> bool:
        def figures(text: Any) -> List[Tuple[float, str]]:
            return [(float(m.group(1)), Text.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", Text.lowered(text))]
        wanted, found = figures(value), figures(form)
        if not wanted:
            return True
        for number, name in found:
            if not any(number == n and Text.same_unit(name, u) for n, u in wanted):
                return False
        return True


class Clock:
    def __init__(self) -> None:
        self.started = monotonic()
        self.marks: List[float] = []
    def now(self) -> float:
        return monotonic() - self.started
    def pace(self) -> float:
        marks = self.marks[-4:]
        if len(marks) < 2:
            return 6.0
        return max(2.0, min(30.0, (marks[-1] - marks[0]) / (len(marks) - 1)))
    def late(self) -> bool:
        return self.now() + 2 * self.pace() > EPISODE_SECONDS
    def overdue(self) -> bool:
        return self.now() > ORDER_NOW


def posted(client: ProxyClient, path: str, body: Dict[str, Any]) -> Tuple[Any, Any]:
    verbose = getattr(client, "post_verbose", None)
    if not callable(verbose):
        return client.post(path, json_data=body), None
    result = verbose(path, json_data=body)
    return getattr(result, "data", None), getattr(result, "error", None)


class Schema:
    @staticmethod
    def fields(properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
        return {"type": "object", "properties": properties, "required": required}
    @staticmethod
    def strings(description: str) -> Dict[str, Any]:
        return {"type": "array", "items": {"type": "string"}, "description": description}


class ModelSeat:
    def __init__(self, name: str, purpose: str, instructions: str, schema: Dict[str, Any], max_tokens: int, engines: Tuple[Engine, ...]) -> None:
        self.name, self.purpose, self.instructions, self.schema = name, purpose, instructions, schema
        self.max_tokens, self.engines = max_tokens, engines
    def tool(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.purpose, "parameters": self.schema}}


class Part:
    def wire(self, clock: Clock, session: StoreSession, catalog: Catalog, desk: ModelDesk, order: Order, reader: Reader, planner: SearchPlanner,
             interviewer: Interviewer, matcher: Matcher, sieve: Sieve, judge: Judge, episode: Episode) -> None:
        self.clock, self.session, self.catalog, self.desk, self.order, self.reader = clock, session, catalog, desk, order, reader
        self.planner, self.interviewer, self.matcher, self.sieve, self.judge, self.episode = planner, interviewer, matcher, sieve, judge, episode


class ModelDesk(Part):
    client = ProxyClient(timeout=45, max_retries=1)
    writer_clients = [ProxyClient(timeout=int(WRITER_SECONDS), max_retries=1) for _ in range(3)]
    def __init__(self, spread: float = 0.5) -> None:
        self.spread = spread
        self.on_chutes = getenv("INFERENCE_PROVIDER", "chutes") != "openrouter"
        self.writers = [engine for engine in WRITERS if not self.on_chutes or engine.chutes]
        self.rounds = 0
        self.waiting: List[threading.Thread] = []
    def out_of_time(self) -> bool:
        return self.clock.now() > MODEL_STOP
    def can_think(self) -> bool:
        return not self.out_of_time()
    def engines(self, seat: ModelSeat, second: bool = False) -> List[Engine]:
        engines = [engine for engine in seat.engines if not self.on_chutes or engine.chutes]
        if self.clock.now() > THINK_STOP:
            engines = [engine for engine in engines if not engine.thinks] or engines
        return engines[1:] if second else engines
    def ask(self, seat: ModelSeat, text: str, second: bool = False) -> Optional[Dict[str, Any]]:
        for engine in self.engines(seat, second):
            if self.out_of_time() or TALK_END - self.clock.now() < 8:
                return None
            parsed = self.ask_engine(engine, seat, text)
            if parsed is not None:
                return parsed
        return None
    def ask_engine(self, engine: Engine, seat: ModelSeat, text: str) -> Optional[Dict[str, Any]]:
        left = TALK_END - self.clock.now()
        if self.out_of_time() or left < 8:
            return None
        ModelDesk.client.timeout = int(min(45.0, left))
        model, settings = engine.route(self.on_chutes)
        body = {"model": model, "max_tokens": seat.max_tokens,
                "messages": [{"role": "system", "content": seat.instructions}, {"role": "user", "content": text}],
                "tools": [seat.tool()], "tool_choice": {"type": "function", "function": {"name": seat.name}}}
        body.update(settings)
        body.setdefault("reasoning", {"enabled": False})
        reply, error = posted(ModelDesk.client, "/inference/chat/completions", body)
        if not isinstance(reply, dict) and (error or {}).get("status") == 429 and TALK_END - self.clock.now() > 20:
            sleep(1.0 + 2.0 * self.spread)
            reply, error = posted(ModelDesk.client, "/inference/chat/completions", body)
        if not isinstance(reply, dict):
            return None
        return self.answer_in(reply, seat.name)
    def ask_agreed(self, seat: ModelSeat, text: str, key: Callable[[Dict[str, Any]], Any],
                   merge: Optional[Callable[[List[Dict[str, Any]]], Optional[Dict[str, Any]]]] = None) -> Optional[Dict[str, Any]]:
        first = self.ask(seat, text)
        voters = [v for v in VOTERS if not self.on_chutes or v.chutes]
        if first is None or not voters or self.clock.now() > VOTE_STOP:
            return first
        readings = [first]
        keys = [key(first)]
        for voter in voters:
            if self.clock.now() > VOTE_STOP or not self.can_think():
                break
            try:
                more = self.ask_engine(voter, seat, text)
            except Exception:
                break
            if more is None:
                continue
            readings.append(more)
            keys.append(key(more))
            if keys[-1] == keys[0]:
                return first
            for i in range(1, len(keys) - 1):
                if keys[i] == keys[-1]:
                    return readings[i]
        if merge is not None and len(readings) >= 3:
            try:
                merged = merge(readings)
            except Exception:
                merged = None
            if merged is not None:
                return merged
        return first
    def ask_all(self, seat: ModelSeat, text: str) -> List[Dict[str, Any]]:
        writers = [FIRST_WRITERS[self.rounds % len(FIRST_WRITERS)]] + self.writers[1:]
        self.rounds += 1
        answers: List[Optional[Dict[str, Any]]] = [None] * len(writers)
        def ask(i: int) -> None:
            answers[i] = self.write(writers[i], ModelDesk.writer_clients[i % len(ModelDesk.writer_clients)], seat, text)
        self.waiting = [t for t in self.waiting if t.is_alive()]
        deadline = min(monotonic() + WRITER_SECONDS + 3.0, self.clock.started + TALK_END)
        threads: List[threading.Thread] = []
        in_turn: List[int] = []
        for i in range(1, len(writers)):
            if self.waiting:
                break
            try:
                thread = threading.Thread(target=ask, args=(i,), daemon=True)
                thread.start()
                threads.append(thread)
            except RuntimeError:
                in_turn.append(i)
        ask(0)
        for i in in_turn:
            if not self.out_of_time():
                ask(i)
        for thread in threads:
            thread.join(max(0.5, deadline - monotonic()))
        self.waiting = [t for t in threads if t.is_alive()]
        got = [a for a in answers if isinstance(a, dict)]
        if not got:
            single = self.ask(seat, text, second=True)
            got = [single] if isinstance(single, dict) else []
        return got
    def write(self, engine: Engine, client: ProxyClient, seat: ModelSeat, text: str) -> Optional[Dict[str, Any]]:
        left = TALK_END - self.clock.now()
        if self.out_of_time() or left < 8:
            return None
        client.timeout = int(min(WRITER_SECONDS, left))
        model, settings = engine.route(self.on_chutes)
        body = {"max_tokens": seat.max_tokens,
                "messages": [{"role": "system", "content": seat.instructions}, {"role": "user", "content": text}],
                "tools": [seat.tool()], "tool_choice": {"type": "function", "function": {"name": seat.name}}, "reasoning": {"enabled": False},
                "model": model, **settings}
        try:
            reply, error = posted(client, "/inference/chat/completions", body)
            if not isinstance(reply, dict) and (error or {}).get("status") == 429 and TALK_END - self.clock.now() > 20:
                sleep(1.0 + 2.0 * self.spread)
                reply, error = posted(client, "/inference/chat/completions", body)
        except Exception:
            return None
        if not isinstance(reply, dict):
            return None
        return self.answer_in(reply, seat.name)
    @staticmethod
    def closed_prefix(raw: str) -> Optional[Dict[str, Any]]:
        cuts = [i + 1 for i, ch in enumerate(raw) if ch in "]}\""]
        for cut in reversed(cuts[-400:]):
            head, stack, quoted, escaped = raw[:cut].rstrip().rstrip(","), [], False, False
            for ch in head:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    quoted = not quoted
                elif not quoted and ch in "[{":
                    stack.append("]" if ch == "[" else "}")
                elif not quoted and ch in "]}" and stack:
                    stack.pop()
            if quoted:
                continue
            try:
                parsed = json.loads(head + "".join(reversed(stack)))
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None
    @staticmethod
    def answer_in(reply: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
        try:
            message = reply["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        for call in message.get("tool_calls") or []:
            raw = (call.get("function") or {}).get("arguments")
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else raw
            except (TypeError, ValueError):
                parsed = ModelDesk.closed_prefix(raw) if isinstance(raw, str) else None
            if isinstance(parsed, dict):
                return parsed
        content = message.get("content")
        if isinstance(content, str) and "{" in content:
            try:
                parsed = json.loads(content[content.index("{"): content.rindex("}") + 1])
            except ValueError:
                return None
            if isinstance(parsed, dict):
                return parsed.get(name) if isinstance(parsed.get(name), dict) else parsed
        return None
    @staticmethod
    def listed_fields(answer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        answer = dict(answer or {})
        for name in ("adds", "drops", "queries"):
            if isinstance(answer.get(name), str):
                try:
                    answer[name] = json.loads(answer[name])
                except ValueError:
                    answer[name] = []
            if not isinstance(answer.get(name), list):
                answer[name] = []
        return answer


class StoreSession(Part):
    client = ProxyClient(timeout=90, max_retries=2)
    def __init__(self, problem: Dict[str, Any]) -> None:
        environment = problem["environment"]
        self.binding = dict(environment["binding"])
        self.spread = self.spread_of(self.binding)
        policy = environment["policy_view"]
        self.query = str(policy.get("query") or "")
        self.tools = {t["function"]["name"]: t["function"] for t in policy.get("tools") or [] if isinstance(t, dict) and "function" in t}
        self.turn_limit = int(policy.get("max_steps") or 20)
        self.group_size = max(1, min(CALLS_PER_TURN, int(policy.get("max_calls_per_turn") or 1)))
        self.task = str(problem.get("problem_id") or problem.get("id") or "task")
        self.turn = 0
        self.over = False
        self.dialogue: List[Dict[str, Any]] = []
    @staticmethod
    def spread_of(binding: Dict[str, Any]) -> float:
        digits = re.sub(r"[^0-9a-f]", "", str(binding.get("session_id") or "").lower())[-2:]
        return int(digits, 16) / 256.0 if digits else 0.5
    @staticmethod
    def call(name: str, **args: Any) -> Dict[str, Any]:
        return {"name": name, "args": {k: v for k, v in args.items() if v is not None}}
    def turns_left(self) -> int:
        return self.turn_limit - self.turn
    def has(self, tool: str) -> bool:
        return tool in self.tools
    def send(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        actions = [a for a in actions if a["name"] in self.tools][: self.group_size]
        if self.clock.now() > STORE_STOP:
            actions = [a for a in actions if a["name"] in ("add_to_cart", "remove_from_cart", "inspect_stock", "inspect_cart", "message", "place_test_order")]
        if not actions or self.over or self.turns_left() <= 0:
            return []
        self.clock.marks.append(monotonic())
        self.turn += 1
        group = f"{self.task}-turn-{self.turn}"
        envelope = {**self.binding, "call_id": group, "idempotency_key": group, "turn": self.turn,
                    "calls": [{"call_id": f"{group}-{i}", "action": {"name": a["name"], "args": a["args"]}} for i, a in enumerate(actions, start=1)]}
        result = self.post(envelope)
        if not isinstance(result, dict) or not isinstance(result.get("calls"), list):
            self.over = True
            self.dialogue.append({"role": "assistant", "content": "", "tool_calls": actions, "environment_result": None})
            return []
        self.dialogue.append({"role": "assistant", "content": "", "tool_calls": actions, "environment_result": result})
        catalog = self.catalog
        observations = []
        known = {key[0] for key in catalog.rows}
        shown = 0
        for action, call in zip(actions, result["calls"]):
            step = call.get("observation") if isinstance(call, dict) else None
            step = step if isinstance(step, dict) else {}
            observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
            if step.get("done") is True:
                self.over = True
            catalog.learn(action, observation)
            observations.append(observation)
            if action["name"] in ("search", "filter"):
                shown += len(observation.get("results") or [])
        if any(a["name"] in ("search", "filter") for a in actions):
            catalog.fresh_log.append((len({key[0] for key in catalog.rows} - known), shown))
        said = result.get("user_message")
        if isinstance(said, dict) and str(said.get("content") or "").strip():
            catalog.heard.append(str(said["content"]).strip())
        return observations
    def post(self, envelope: Dict[str, Any]) -> Any:
        result = None
        for pause in SEND_PAUSES + (None,):
            left = SEND_END - self.clock.now()
            StoreSession.client.timeout = int(max(10.0, min(90.0, left)))
            StoreSession.client.max_retries = 2 if left > 120 else 1
            result, error = posted(StoreSession.client, "/environment/call", envelope)
            if isinstance(result, dict) and isinstance(result.get("calls"), list):
                return result
            if pause is None or (error or {}).get("status") not in BUSY:
                return result
            if SEND_END - self.clock.now() < pause + 20.0:
                return result
            sleep(pause)
        return result


class Catalog(Part):
    def __init__(self) -> None:
        self.rows: Dict[Key, Dict[str, Any]] = {}
        self.row_order: List[Key] = []
        self.found_by: Dict[str, Set[str]] = {}
        self.shown_at: Dict[Key, int] = {}
        self.shelf: Dict[Key, str] = {}
        self.pages: Dict[str, Dict[str, Any]] = {}
        self.page_writes = 0
        self.stock: Dict[Key, Dict[str, Any]] = {}
        self.cart: List[Key] = []
        self.heard: List[str] = []
        self.fresh_log: List[Tuple[int, int]] = []
        self.shelved_cache: Tuple[Any, ...] = ((), False)
        self.shelf_cache: Tuple[int, str] = (-1, "")
    def note_stock(self, product_id: Any, sku: Any, price: Any, in_stock: Any) -> None:
        if product_id is None or sku is None:
            return
        entry = self.stock.setdefault((str(product_id), str(sku)), {})
        entry["turn"] = self.session.turn
        if isinstance(price, (int, float)) and not isinstance(price, bool):
            entry["price"] = float(price)
            entry.setdefault("first", float(price))
        if isinstance(in_stock, bool):
            entry["in_stock"] = in_stock
    def learn(self, action: Dict[str, Any], observation: Dict[str, Any]) -> None:
        name = action["name"]
        if name in ("search", "filter"):
            query = Text.lowered(action["args"].get("query")) if name == "search" else ""
            for row in observation.get("results") or []:
                if isinstance(row, dict) and row.get("product_id") and row.get("sku"):
                    key = (str(row["product_id"]), str(row["sku"]))
                    if query:
                        self.found_by.setdefault(key[0], set()).add(query)
                    if key not in self.rows:
                        self.rows[key] = row
                        self.row_order.append(key)
                        self.shown_at[key] = self.session.turn
                    if name == "filter" and action["args"].get("category"):
                        self.shelf.setdefault(key, str(action["args"]["category"]))
                    self.note_stock(key[0], key[1], row.get("price"), row.get("in_stock"))
        elif name in ("view", "compare"):
            for page in ([observation] if name == "view" else observation.get("compared") or []):
                if isinstance(page, dict) and page.get("product_id"):
                    self.pages[str(page["product_id"])] = page
                    self.page_writes += 1
                    for variant in page.get("variants") or []:
                        if isinstance(variant, dict):
                            self.note_stock(page["product_id"], variant.get("sku"), variant.get("price"), variant.get("in_stock"))
        elif name == "inspect_stock":
            self.note_stock(observation.get("product_id"), observation.get("sku"), observation.get("price"), observation.get("in_stock"))
        elif name == "add_to_cart":
            added = observation.get("added") if isinstance(observation.get("added"), dict) else None
            if added:
                key = (str(added.get("product_id")), str(added.get("sku")))
                if key not in self.cart:
                    self.cart.append(key)
                self.note_stock(key[0], key[1], added.get("price"), added.get("in_stock"))
    def named_shelf(self) -> str:
        pages = self.pages
        if self.shelf_cache[0] == len(pages):
            return self.shelf_cache[1]
        request = f" {Text.label_key(self.order.request)} "
        said = f" {Text.label_key(self.order.shelf_said)} " if self.order.shelf_said else ""
        seen: Dict[str, int] = {}
        told: Dict[str, int] = {}
        for page in pages.values():
            path = [str(label) for label in page.get("category_path") or []]
            if path and Text.label_key(path[-1]) and f" {Text.label_key(path[-1])} " in request:
                seen[path[-1]] = seen.get(path[-1], 0) + 1
            if path and said and Text.label_key(path[-1]) and f" {Text.label_key(path[-1])} " in said:
                told[path[-1]] = told.get(path[-1], 0) + 1
        own = {Text.label_key(self.order.category), Text.label_key(self.order.item)} - {""}
        best = max(seen, key=lambda label: (len(Text.label_key(label).split()), Text.label_key(label) in own, seen[label]), default="")
        if told:
            best = max(told, key=lambda label: (len(Text.label_key(label).split()), told[label]))
        self.shelf_cache = (len(pages), best)
        return best
    def in_category(self, page: Dict[str, Any]) -> bool:
        named = self.named_shelf()
        wanted = {Text.label_key(named)} if named else {Text.label_key(self.order.category), Text.label_key(self.order.item)} - {""}
        labels = {Text.label_key(label) for label in page.get("category_path") or []}
        return bool(wanted & labels)
    def any_shelved(self) -> bool:
        stamp = (self.page_writes, self.named_shelf(), self.order.category, self.order.item)
        if self.shelved_cache[0] != stamp:
            self.shelved_cache = (stamp, any(self.in_category(p) for p in self.pages.values()))
        return self.shelved_cache[1]
    def variant_of(self, key: Key) -> Dict[str, Any]:
        page = self.pages.get(key[0]) or {}
        return next((v for v in page.get("variants") or [] if str(v.get("sku")) == key[1]), {})


class Order(Part):
    def __init__(self, request: str) -> None:
        self.request = request
        self.beside: List[str] = []
        self.backed_by = ""
        self.item = ""
        self.category = ""
        self.requirements: List[Dict[str, Any]] = []
        self.budget: Optional[float] = None
        self.brand = ""
        self.listed: List[Key] = []
        self.priority = ""
        self.shelf_said = ""
        self.pair_fallback = ""
        self.pair_chosen = False
        self.told_to_ask = False
        self.pair_question = ""
        self.pair_wishes: List[str] = []
        self.pair_brand = ""
        self.brand_chosen = False
        self.set_aside: List[Dict[str, Any]] = []
        self.changed_at = 0
        self.unanswered: List[str] = []
        self.purposes: List[str] = []
        self.invited = False
        self.fresh: Dict[str, List[str]] = {}
        self.fresh_limit: Set[str] = set()
        self.took_back: Dict[str, str] = {}
        self.opening: List[Dict[str, Any]] = []
        self.opening_budget: Optional[float] = None
        self.changed_early = False
        self.moved_on = False
        self.look_at = 0
        self.dropped: Set[str] = set()
    @staticmethod
    def requirement(raw: Dict[str, Any]) -> Dict[str, Any]:
        def texts(name: str) -> List[str]:
            return [str(w).strip() for w in raw.get(name) or [] if str(w).strip()][:6] if isinstance(raw.get(name), list) else []
        value = re.sub(r"(\d)\s*(?:\"|”|″|'')", r'\1"', str(raw.get("value") or "").strip())
        value = re.sub(r"(\d)-([a-z])", r"\1 \2", value)
        name = Text.lowered(raw.get("property"))
        if re.fullmatch(r"\d+(?:\.\d+)?", value) and re.fullmatch(r"[a-z]{1,3}", name.strip()):
            value = f"{value} {name.strip()}"
        filed = not re.search(r"\d", value) and not re.search(r"colou?r|material", name)
        written = texts("written")
        alike = texts("alike")
        if not re.search(r"\d", value) and not re.search(r"colou?r", name):
            parts = [part.strip() for part in re.split(r"\s+(?:and|&)\s+", value) if part.strip()]
            if len(parts) >= 2 and all(len(Text.word_list(part)) <= 3 for part in parts):
                alike += [part for part in parts if Text.closed_up(part) not in {Text.closed_up(x) for x in alike}]
        spelt = [w for w in alike if Text.one_letter_apart(Text.lowered(w), Text.lowered(value))]
        written += [w for w in spelt if Text.closed_up(w) not in {Text.closed_up(x) for x in written}]
        return {"filed": filed, "property": str(raw.get("property") or "").strip(), "value": value,
                "written": written, "alike": [w for w in alike if w not in spelt]}
    def state_text(self) -> str:
        lines = [f"Request: {self.request}", f"Item: {self.item}",
                 "Requirements: " + ("; ".join(f"{r['property']} = {r['value']}" for r in self.requirements) or "(none)"),
                 f"Budget: {self.budget if self.budget is not None else 'none'}"]
        if self.changed_at:
            lines.insert(1, "The shopper has changed some of this since: where the request and the lines below differ, only the "
                            "requirements and the budget below hold now.")
        if self.priority:
            lines.append(f"Priority between fitting items: {self.priority}")
        return "\n".join(lines)
    def holds_now(self, said: str) -> Tuple[List[Dict[str, Any]], bool]:
        lines = [line for line in list(self.fresh) + list(self.fresh_limit) if line and line in said]
        values = {Text.lowered(v) for line in lines for v in self.fresh.get(line, [])}
        return [r for r in self.requirements if Text.lowered(r["value"]) in values], any(line in self.fresh_limit for line in lines)
    def is_wish(self, value: str) -> bool:
        words = {Text.singular(w) for w in Text.word_list(value)}
        return bool(words) and any(words <= {Text.singular(w) for w in Text.word_list(wish)} for wish in self.pair_wishes)


REQUEST_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in two or three sentences: which words of the request give the item, each requirement, the priority and each purpose, and what you left out of the record and why."},
    "item": {"type": "string", "description": "The shopper's own words for the kind of thing, copied from the request exactly as written, without the article and without the requirements."},
    "category": {"type": "string", "description": "A store category, only when the shopper names one as a category; otherwise empty."},
    "requirements": {"type": "array", "description": "Each property the item itself must have, stated in the request.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The name of the property in the shopper's words (colour, material, pattern, style, capacity, ...)."},
            "value": {"type": "string", "description": "The required value, as the shopper wrote it, with its unit."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: the other spellings of the word (another country's spelling; joined, hyphenated or spaced); for an amount, the same amount in the other units and notations sellers use for this measure, converted exactly (with the decimals the conversion gives: a rounded figure is another amount), a whole number first when the conversion gives one. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value: a plain synonym, a technical or trade word, an abbreviation, or a narrower kind that is still the value (a specific kind of the wanted material or type); never a part of it, a wider kind, or something that only goes with it. Empty when there is none."}},
        "required": ["property", "value"]}},
    "budget": {"type": "number", "description": "The most the shopper will pay; 0 when none is named."},
    "brand": {"type": "string", "description": "A brand the shopper requires; empty when none, and empty when a brand is only preferred or a first choice."},
    "priority": {"type": "string", "description": "What the shopper wants the most or the least of among the items that fit, in the shopper's words (for example the lowest price). Most requests name one: read the request again before leaving this empty."},
    "queries": Schema.strings("Exactly six short catalogue searches (two to five words) that could return the item: the item words with the required values, with a synonym or another spelling of a value where one exists."),
    "priority_queries": Schema.strings("When a priority is given: up to four more searches that reach the listings standing best by it; otherwise empty."),
    "doubted_pair": {"type": "array", "items": {"type": "string"}, "maxItems": 2,
                     "description": "When the shopper wishes for two things and doubts one item can have both: the two wishes, each as its value with what it describes (for example 'a leather strap', 'the Acme brand'). Empty otherwise."},
    "purposes": {"type": "array", "items": {"type": "string"}, "maxItems": 3,
                 "description": "Each use or recipient the request gives for the item, one entry each, in the shopper's words. Empty when it gives none."},
    "invitation": {"type": "string", "description": "The request's exact words that tell the assistant to put a question to the shopper before deciding (to ask, check or confirm with the shopper whether the purpose changes what the item needs); empty when the request holds no such instruction."},
    "look_at": {"type": "integer", "description": "When the request tells the assistant to open, compare or check at least some number of fitting options before choosing: that number; otherwise 0."},
    "backed_by": {"type": "string", "description": "When the request asks for the order to be backed or justified with facts seen on the listing: the one detail of the item it names outright besides the price and the stock (not one it offers only as an example), in the request's words; empty otherwise."},
    "open_properties": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
                        "description": "Properties of this kind of item that the request does not settle and a shopper could still have a requirement about, the likeliest first, each in one to three plain words."},
}, ["reason", "item", "requirements", "budget", "priority", "queries", "doubted_pair", "purposes", "invitation", "look_at", "backed_by", "open_properties"])

REQUEST_INSTRUCTIONS = """You are the order writer of a shopping assistant. A shopper has written one request for one item in an online store. You turn the request into the order record the assistant shops by, and fill the plan tool from the request's own words. The shopper trusts the assistant only if the item ordered meets every one of their requirements, and one requirement read wrongly, or left out, loses that trust. Read exactly what the words say; never assume, guess or fill in what they do not say.

WHAT YOU ARE GIVEN
The request, word for word.

WHAT YOU RETURN
The plan tool: reason first, then the item, the requirements, the budget, the brand, the priority, the doubted pair, the purposes, the invitation, look_at, backed_by, the properties still open and the searches.

HOW TO READ THE REQUEST - work through these steps in order.

Step 1 - The item.
- item: the words the request uses for the kind of thing, copied completely and exactly as written, keeping a leading word such as 'other' and any '&' or '/' (the store files its listings under such names). Never shorten them, and never put a broader or narrower kind in their place. When the name joins several kinds ('<kind> & <kind>', '<kind>/<kind>'), a detail that names one of those kinds is a requirement that picks the kind, never the item said again.
- category: a store category only when the request names one as a category; otherwise empty.

Step 2 - The requirements. Every detail the request gives about the item itself is a requirement. Go through these kinds of detail one by one and take every one the request gives:
 a. what the item is made of (its material, or the material of a named part of it);
 b. its colour;
 c. an amount with its unit: how much it holds, how big, long, wide, thick or heavy it is, how many pieces come together, how strong or powerful it is;
 d. its pattern, shape, style, finish, kind or type (a kind or type named for the item is a requirement);
 e. an occasion or use named together with the item as part of what it is;
 f. the make, model, size or thing the item must fit, match or work with;
 g. anything the item must include or contain.
For each requirement:
- property: the name of the property in the shopper's words, every word of it as they said it (a store files its facts under names of several words, and a name cut to one general word reads the wrong fact); when the request gives only the value, the plain name of that kind of detail (a colour word is a colour, what it is made of is its material, an amount is the measure its unit tells).
- value: the value in the fewest words - the word or the amount itself, an amount with its unit written in letters - without the property's name, without the item's name and without words such as must, required or essential.
- written: the same value written another way. For a word: its other spellings - another country's spelling of it, and the word joined, hyphenated or spaced as sellers write it. For an amount: the same amount in each other unit or notation sellers of this kind of item use for that measure, converted exactly - with the decimals the conversion gives, because a rounded figure is another amount and a listing that writes it does not state this one; when the amount converts to a whole number in another unit, that form comes first; and when sellers write the unit as a mark right after the figure, the amount with that mark as well.
- alike: other words sellers use for the same value: a plain synonym, a technical or trade word for it, an abbreviation sellers use, or a narrower kind that is still the value (a specific kind of the wanted material, type or style). Never a part of it, a wider kind, a neighbouring value or something that only goes with it. Leave either empty when there is none.
Rules:
- One phrase of the request gives one requirement: never split one phrase into two, never join two phrases into one.
- Not requirements: the budget, being in stock, the store's category, the priority, a brand that is only preferred, and why the shopper is buying (it comes in a sentence of its own). A value the request sets for a property of the item is a requirement whatever that property is called: a property's name never makes its value a mere wish. A sentence that says what the item has to match, fit, follow or go with (something the shopper already owns, uses or has planned) gives a purpose, not a requirement: it names no property and no value of the item.
- A request for one kind of thing nearly always carries at least one requirement: read it again before leaving the list empty.

Step 3 - Two doubted wishes (doubted_pair). A request sometimes names two things the shopper would like on top of what the item must have - usually a value of some property and a brand - and doubts, or leaves open, that one item offers both: a doubt said outright, or the second wish added as something further that may not be possible. The shopper will say which matters more when asked. Give the two wishes, each as its value with what it describes (the <value> <property>; the <maker> brand). The two wishes are not requirements: leave them out of requirements. Leave doubted_pair empty when the request names fewer than two such wishes: a single thing the shopper would merely like (one brand only preferred) is neither a requirement nor a pair, and two words of the item's own name are not two wishes. The priority is never a wish: a preferred brand beside the priority, or a requirement beside the priority, makes no pair.

Step 4 - The budget and the brand.
- budget: the most the shopper will pay, as a plain number; 0 when none is named.
- brand: a brand the shopper requires. A brand that is only preferred, or wanted only when there is one, leaves it empty.

Step 5 - The priority.
- priority: what the shopper wants the most or the least of among the items that fit - the most or the least of some quantity the items differ in, or the lowest price - in the shopper's words. Most requests name one, in a sentence of its own or beside the budget: never leave it out when the request has one. The spending limit itself is no priority: a request that only caps the price names none.
- priority_queries: when there is a priority, up to four searches that reach the listings standing best by it: the item words with the kind of amount such listings write.

Step 6 - Purposes and instructions.
- purposes: every use or recipient the request gives (what the item is for, whom it is for). Each can carry a need the shopper has not stated yet. A request often gives two - in two sentences, or in one sentence joined by a linking word, the second sometimes only as a further clause: give each as its own entry, in the shopper's own words for it, without the linking words, as a phrase that can follow the word 'for'. One purpose is one entry: never cut one purpose in two. A sentence that says what the item is meant for, or whom it is for, gives a purpose however it begins - leave none out. A clause that only voices a doubt, or a condition of the purchase itself, is not a purpose: a purpose says what the item is for or whom it is for. Two sentences that each give a purpose are two entries even when they sound alike (<a purpose>; <the same kind of purpose said in other words>): each can hold a need of its own.
- invitation: the request's exact words that tell the assistant to put a question to the shopper before deciding (to ask, check or confirm with the shopper whether the purpose changes what the item needs). A sentence that only gives the purpose, says the purpose may matter, or says what the item must be holds no such instruction: leave it empty.
- look_at: when the request tells the assistant to open, compare, check or go through at least some number of fitting options before choosing: that number; otherwise 0.
- backed_by: when the request asks for the order to be backed or justified with facts seen on the listing: the one detail of the item it names outright besides the price and the stock, in the request's words. Never the price or the stock, and never something it offers only as an example. Otherwise empty.

Step 7 - What is still open (open_properties). What the request leaves open about the item itself, in the order to ask about it. A shopper who holds one requirement back holds back something plain, whatever the kind of item: do not judge whether it sounds usual for this item.
- When a stated purpose itself points to a property of the item (how it has to look, what it has to hold, weigh or work with for that purpose), that property comes first.
- After it the list always runs in this order, leaving out only what the request already settles: the colour; then which type of the item it is, written as 'type of' and the item's main word or two (type of <item word>; never with 'and' or '&'); then the piece count (how many pieces come together); then the material; then the amounts this kind of item is sold by, each named by the measure itself and never by its unit. Nothing else goes ahead of these.
- Name each in one to four plain words; never a loose word that leaves open which measure or which kind is meant (a bare 'size' does).
- Leave out the quantity named as the priority, the brand, the budget, the category and stock.

Step 8 - Searches (queries). Exactly six searches as a shopper types them, two to five words each, built from the item words and the required values: the item words with one value, with another value, with a value's other spelling or synonym, with two values together. Write an amount the way listings do (the number joined to its unit).

reason: written before every other field - two or three sentences on which words gave the item, each requirement, the priority and each purpose, and what you left out and why."""

SEAT_REQUEST = ModelSeat('plan', "The shopper's request, written down for shopping.", REQUEST_INSTRUCTIONS, REQUEST_SCHEMA, 2048, (GLM, GLM, LUNA))

LINE_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one or two sentences: what the line says, and how you read it into what it adds, takes back or puts in place of an earlier value."},
    "adds": {"type": "array", "description": "Each requirement of the item the line states that was not known before.", "items": {
        "type": "object", "properties": {
            "property": {"type": "string", "description": "The property's name as the shopper said it, with every word of that name (the material of one part is named with the part)."},
            "value": {"type": "string", "description": "The value alone, as the shopper said it: an amount with its unit; never the property's own name, and never a word that only says it is needed."},
            "written": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "The same value written another way: the other spellings of the word (another country's spelling; joined, hyphenated or spaced); for an amount, the same amount in the other units and notations sellers use for this measure, converted exactly (with the decimals the conversion gives: a rounded figure is another amount), a whole number first when the conversion gives one. Empty when there is none."},
            "alike": {"type": "array", "items": {"type": "string"}, "maxItems": 6, "description": "Other words sellers use for the same value: a plain synonym, a technical or trade word, an abbreviation, or a narrower kind that is still the value (a specific kind of the wanted material or type); never a part of it, a wider kind, or something that only goes with it. Empty when there is none."},
            "keeps_earlier": {"type": "boolean", "description": "True when the line adds this value on top of the value already required for the same property, so that both must hold; false when it puts this value in place of an earlier one, or the property had none."}},
        "required": ["property", "value", "written", "alike", "keeps_earlier"]}},
    "drops": Schema.strings("Properties whose earlier requirement the line withdraws or puts another value in place of; never a property whose earlier value the line keeps beside a new one."),
    "budget": {"type": "number", "description": "A new spending limit the line states; 0 when it states none."},
    "priority": {"type": "string", "description": "What the line says matters most between fitting items; empty when nothing."},
    "asks": {"type": "boolean", "description": "True when the line asks the assistant something or expects an answer."},
    "about": {"type": "string", "description": "Only when the assistant had just asked about several properties and the line gives a value: the one of them that value belongs to, copied exactly as the assistant named it. Empty when it belongs to none of them or nothing was asked."},
    "about_change": {"type": "boolean", "description": "True when the line only reacts to the stock or the price of the chosen item having changed."},
    "queries": Schema.strings("Up to six short catalogue searches that use what the line added (try a synonym or another spelling of the value too); empty when it added nothing."),
}, ["reason", "adds", "drops", "budget", "asks", "queries"])

LINE_INSTRUCTIONS = """You are the order keeper of a shopping assistant. A shopper is being served, and the order record below holds what they want now. A new line from the shopper has arrived: fill the hear tool with exactly what this line changes in the record. The shopper trusts the assistant only if the item ordered meets every one of their requirements; one requirement read wrongly, or one dropped that still holds, loses that trust. Read exactly what the words say; never assume, guess or fill in what they do not say.

WHAT YOU ARE GIVEN
The shopper's request, the item, the requirements now in force (property = value), the budget and the priority between fitting items; sometimes the question the assistant had just asked; then the shopper's new line.

WHAT YOU RETURN
The hear tool: reason first, then adds, drops, budget, priority, asks, about, about_change and queries.

STEP 1 - WHAT KIND OF LINE IT IS. A line can be of more than one kind: then do every part. A line can hold several sentences: read them in order; about the same property, a later one overrides an earlier one.
 Case A - An answer. The assistant asked about something and the line gives a value: that is a requirement, add it. An answer takes nothing back unless it says so in words such as instead, no longer, forget, change, switch, replace. A second value for a property already required is a second requirement: add it with keeps_earlier true.
 Case B - A replacement. The line puts a new value in place of an earlier one, with words such as instead of, forget, no longer, switch, change, replace, rather than. Add the new value under the property it belongs to, and list that property under drops. Both halves are needed: when the line names the new value, never drop without adding it. The new requirement takes the line's own words for the property; when they differ from the name the record uses, list the record's name under drops.
 Case C - An addition. The line says the item must also have or be something: add it and drop nothing. When it is a further value for a property already required, keeps_earlier is true.
 Case D - A withdrawal. A value taken back with nothing in its place: only drop it.
 Case E - A new spending limit. The line names a new budget. Such a line often says again what the shopper still needs: a value already required, said again, is not new - it adds nothing and drops nothing. A value the line says STILL holds is never a drop, whatever else the line says, and it replaces nothing: when an earlier line already put another value in its place, the line is only out of date and changes nothing at all. But a value the line says is needed that is NOT among the requirements in force is new: add it, even when it is worded as something that still holds or as a question. A line that names a limit and then asks something is Case E with asks true.
 Case F - A choice between two wishes. The assistant had asked which of two wishes matters more: add the chosen wish as a requirement and drop nothing (the assistant sets the other wish aside itself) - even when the record seems to hold it already: the record sets a doubted wish aside until the shopper chooses, so the chosen wish is always new. A chosen maker is added with the property brand. An answer that keeps both wishes adds both; one that names neither wish and gives some other detail adds that detail.
 Case G - The stock or the price. The line reacts to the chosen item having sold out or changed price: about_change is true. It adds nothing unless it also states a requirement.
 Case H - Nothing new. The line only confirms, thanks, asks the assistant to go on, or repeats the spending limit and the store's category for the item: nothing changes. When such a line ends with a question - it asks to have the item or a value confirmed - nothing changes and asks is true.

STEP 2 - EACH REQUIREMENT THE LINE STATES. A requirement is one property of the item with the value it must have. Sentences put the two parts in either order and in many shapes: read each sentence for its grammar, not for the order of its words.
 2a. A value said to be of some type or kind: that type or kind, with all its words, is the property, and the thing said to be of it is the value - never the other way round.
 2b. A value said to be included, present, contained or required: the value is the thing included. Words such as included, present, required, essential or needed are never the value. The words naming what kind of thing it is (often just before the value) are the property.
 2c. The item, or a named part of it, said to have or include a value: the property is that item's or part's own name for the kind of thing included (or the part's name), never a bare word such as includes.
 2d. The value may come before the property's name or after it, joined by a word such as as, for or of: the value is the thing the item must be or have, and the property is what kind of detail it is.

STEP 3 - THE PROPERTY. Use the shopper's own full name for the property, with every word of that name (the line's words for a kind, a part or a measure, together with its 'type' or 'material' when the line says it). Never cut the name to one general word (type, kind, size, style, fit, feature, includes) when the line names it more fully, and never put a name of your own in its place. When the line gives only the value, name the property by what the value is: a colour word is a colour, what a thing is made of is its material, an amount is the measure its unit tells. The property never repeats the value's own words.

STEP 4 - THE VALUE. The value alone, in the fewest words: no verb (must be, has to be, needs, include, contain, come in), no article, no word of the property's own name, nothing about why it is wanted. Keep every word that names the value, also a word it shares with the item's name (a kind word followed by the item word names one kind of the item: keep both words). An amount keeps its number and its unit as the shopper wrote them; when the line gives only a number for a property already required with a unit, the value takes that unit.

STEP 5 - written AND alike.
- written: other ways sellers write the same value. For a word: another country's spelling of it, and the word joined, hyphenated or spaced. For an amount: the same amount in each other unit or notation sellers of this kind of item use for that measure, converted exactly - with the decimals the conversion gives, as a rounded figure is another amount - the whole-number form first when the conversion gives one; a standard figure by its short name or by its full figures.
- alike: other words sellers use for the same value: a plain synonym, a technical or trade word for it, an abbreviation, or a narrower kind that is still the value (a specific kind of the wanted material, type or style). Never a part of it, a wider kind, a neighbouring value or something that only goes with it.
- Leave either empty when there is none.

STEP 6 - WHAT IS NEVER A REQUIREMENT. Being in stock, available or ready; the budget; the store's category; why the shopper buys or whom it is for. Leave these out of adds.

STEP 7 - drops. Only properties whose earlier value the line itself takes back or replaces, named as the record names them. Never a property the line does not speak of, and never one because the line is about something else.

STEP 8 - THE OTHER FIELDS.
- budget: a new spending limit the line itself states; 0 when it states none. A price the line mentions for an item, or a change in a price, is not a budget.
- priority: what the line says matters most between fitting items; empty when it says nothing of that.
- asks: true when the line puts a question to the assistant or asks to have something confirmed; a remark that needs no answer is false.
- about: only when the assistant had just asked about several properties and the line gives a value: the one of them that value belongs to, copied as the assistant named it.
- about_change: true when the line reacts to the stock or the price of the chosen item having changed.
- queries: searches of two to five words that combine the item's words with each value just added; empty when nothing was added.
- reason: written before every other field, in one or two sentences: which case or cases the line is, and how you read it into adds and drops."""

SEAT_LINE = ModelSeat('hear', "What the shopper's new line changes.", LINE_INSTRUCTIONS, LINE_SCHEMA, 2048, (GLM, GLM, LUNA))


class Reader(Part):
    def __init__(self) -> None:
        self.pending_line = ""
        self.reply_sent: Optional[Tuple[int, str]] = None
        self.replies_again = 0
    @staticmethod
    def plan_key(answer: Dict[str, Any]) -> Any:
        answer = answer or {}
        requirements = frozenset((Text.lowered(r.get("property")), Text.closed_up(r.get("value"))) for r in answer.get("requirements") or []
                                 if isinstance(r, dict) and str(r.get("value") or "").strip())
        return (Text.lowered(answer.get("item")), requirements, Text.money(answer.get("budget")), Text.lowered(answer.get("brand")),
                frozenset(Text.content_words(str(answer.get("priority") or ""))), frozenset(Text.lowered(w) for w in answer.get("doubted_pair") or [] if str(w).strip()),
                Text.lowered(answer.get("category")), len([p for p in answer.get("purposes") or [] if str(p).strip()]))
    def wish_add(self, wish: str) -> Dict[str, Any]:
        words = Text.word_list(wish)
        if len(words) >= 2 and words[-1] == "brand":
            return {"property": "brand", "value": " ".join(wish.split()[:-1]).strip(" ,.")}
        for r in self.order.set_aside:
            if Text.lowered(r["value"]) in Text.lowered(wish):
                return {"property": r["property"], "value": r["value"], "written": list(r.get("written") or []), "alike": list(r.get("alike") or [])}
        return {"property": "requirement", "value": wish}
    def read_request(self) -> str:
        request = self.order.request
        plan = self.desk.ask_agreed(SEAT_REQUEST, f"The shopper's request:\n{request}", Reader.plan_key) or {}
        if not [r for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]:
            again = self.desk.ask(SEAT_REQUEST, f"The shopper's request:\n{request}", second=True) or {}
            if again.get("requirements"):
                plan = {**plan, **{k: v for k, v in again.items() if v}} if plan else again
        self.order.item = str(plan.get("item") or "").strip()
        self.order.category = str(plan.get("category") or "").strip()
        if self.order.category and not set(Text.word_list(self.order.category)) <= set(Text.word_list(request)):
            self.order.category = ""
        self.order.requirements = [self.order.requirement(r) for r in plan.get("requirements") or [] if isinstance(r, dict) and str(r.get("value") or "").strip()]
        self.order.requirements = [dict(r, stated=True) for r in self.order.requirements if not NOT_A_PROPERTY.search(Text.lowered(f"{r['property']} {r['value']}"))]
        uses = [Text.lowered(" ".join(str(w).split())) for w in plan.get("purposes") or [] if str(w).strip()]
        whole = Text.lowered(" ".join(request.split()))
        rest = whole
        for use in uses:
            rest = rest.replace(use, " ")
        def same_root(a: str, b: str) -> bool:
            shared = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            return shared >= 5 and shared * 10 >= 7 * min(len(a), len(b))
        def only_the_use(value: str) -> bool:
            words = [w for w in Text.word_list(Text.lowered(value)) if len(w) >= 4]
            if not words or re.search(r"\d", value) or rest == whole:
                return False
            in_use = all(any(w == u or same_root(w, u) for use in uses for u in Text.word_list(use)) for w in words)
            return in_use and not any(w == u or same_root(w, u) for w in words for u in Text.word_list(rest))
        self.order.requirements = [r for r in self.order.requirements if not only_the_use(r["value"])]
        def leads_a_use(value: str) -> bool:
            words = Text.word_list(Text.lowered(value))
            if len(words) != 1 or len(words[0]) < 4 or re.search(r"\d", value):
                return False
            leads = [Text.word_list(use)[0] for use in uses if Text.word_list(use)]
            return any(lead.endswith("ing") and (words[0] == lead or same_root(words[0], lead)) for lead in leads)
        self.order.requirements = [r for r in self.order.requirements if not leads_a_use(r["value"])]
        purpose_words = [Text.content_words(use) for use in uses]
        self.order.requirements = [r for r in self.order.requirements if re.search(r"\d", r["value"]) or len(Text.content_words(r["value"])) < 2
                                   or not any(Text.content_words(r["value"]) <= words for words in purpose_words if words)]
        self.order.beside: List[str] = []
        for r in list(self.order.requirements):
            if len(r["value"].split()) >= 3 and re.search(r"\b(?:i|me|my|we|our|us)\b", Text.lowered(r["value"])):
                self.order.requirements.remove(r)
                self.order.beside.append(r["value"])
        for r in list(self.order.requirements):
            use = self.beside_a_purpose(request, r, uses)
            if use:
                self.order.requirements.remove(r)
                self.order.beside.append(use)
        self.order.budget = Text.money(plan.get("budget"))
        if self.order.budget is None:
            amounts = [float(m.replace(",", "")) for m in re.findall(r"(\d[\d,]*\.\d{2})", request)]
            self.order.budget = max(amounts) if amounts else None
        self.order.brand = str(plan.get("brand") or "").strip()
        self.order.listed = [(m.group(1), m.group(2)) for m in re.finditer(r"(\d{6,})::([\w-]+)", request)]
        self.order.priority = self.grounded_priority(plan.get("priority"), request)
        if self.order.priority:
            most = Text.name_words(self.order.priority)
            self.order.requirements = [r for r in self.order.requirements if re.search(r"\d", r["value"])
                                 or not (Text.name_words(r["value"]) - {"the", "a", "an"} <= most and Text.name_words(r["property"]) & most)]
        self.planner.add_queries(self.planner.value_queries() + list(plan.get("queries") or []))
        if self.order.priority:
            self.planner.add_queries([q for q in plan.get("priority_queries") or [] if isinstance(q, str)][:4])
        pair = [" ".join(str(w).split()) for w in plan.get("doubted_pair") or [] if str(w).strip()][:2]
        purposes = ([" ".join(str(w).split()) for w in plan.get("purposes") or [] if str(w).strip()] + self.order.beside)[:3]
        if not self.planner.queries:
            self.planner.add_queries([" ".join(Text.word_list(request)[:5])])
        purposes = [p for i, p in enumerate(purposes) if not any(i != j and self.within(p, q) and (not self.within(q, p) or j < i)
                                                                for j, q in enumerate(purposes))]
        self.order.purposes = purposes
        try:
            self.order.look_at = max(0, min(8, int(plan.get("look_at") or 0)))
        except (TypeError, ValueError):
            self.order.look_at = 0
        backed = " ".join(str(plan.get("backed_by") or "").split())[:60]
        if Text.content_words(backed) and Text.content_words(backed) <= Text.content_words(request) | Text.stems(request):
            self.order.backed_by = backed
        quoted = " ".join(str(plan.get("invitation") or "").split()).strip(" .\"'")
        asking = [w for w in Text.word_list(Text.lowered(quoted)) if len(w) >= 4]
        of_a_purpose = sum(any(same_root(w, u) or w == u for p in purposes for u in Text.word_list(Text.lowered(p))) for w in asking)
        apart = (len(quoted.split()) >= 2 and Text.plain_quotes(quoted) in Text.plain_quotes(request)
                 and Text.lowered(quoted.split()[0]) not in ("who", "whom", "whose", "which", "that")
                 and not any(self.within(quoted, p) or self.within(p, quoted) for p in purposes)
                 and of_a_purpose * 2 < max(1, len(asking)))
        asks_me = r"\b(?:ask|check|confirm|consult|tell)\b[^.?!]{0,24}\bme\b|\blet me know\b|\bif that\b|\bjust say\b"
        self.order.told_to_ask = bool(re.search(asks_me, Text.plain_quotes(request)))
        of_the_pair = any(Text.content_words(w) and Text.content_words(w) <= Text.content_words(quoted) for w in pair)
        if apart and len(purposes) < 2 and not re.search(asks_me, Text.plain_quotes(quoted)) and not of_the_pair:
            words = quoted.split()
            start = next((i for i, w in enumerate(words) if Text.lowered(w) in ("a", "an", "the", "my", "our") or (Text.lowered(w).endswith("ing") and len(w) > 5)), None)
            if start is not None and len(words) - start >= 3:
                purposes = purposes + [" ".join(words[start:])]
                self.order.purposes = purposes
                apart = False
        if len(purposes) < 2:
            purposes = purposes + self.unread_purposes(request, purposes, pair)[: 2 - len(purposes)]
            self.order.purposes = purposes
        self.order.invited = apart and not purposes
        self.interviewer.needs_expected = max(1, min(2, len(purposes)))
        self.interviewer.plain_pending = self.order.told_to_ask
        self.interviewer.queue_properties(["colour"] + [w for w in plan.get("open_properties") or [] if isinstance(w, str)])
        self.interviewer.to_ask.sort(key=lambda name: not re.search(r"colou?r", name))
        said = set(Text.word_list(request))
        wishes = [[x for x in Text.word_list(w) if x not in ("the", "a", "an", "of", "as", "its", "it", "to", "be", "with")] for w in pair]
        capitals = {w.lower() for w in re.findall(r"\b[A-Z][A-Z0-9]+\b", request)}
        spoken = all((words and sum(x in said for x in words) * 2 >= len(words)
                      and any(x in said and (len(x) >= 3 or x[0].isdigit() or x in capitals) for x in words))
                     or any(Text.lowered(r["value"]) in Text.lowered(wish) and Text.says_value(request, r["value"]) for r in self.order.requirements)
                     for words, wish in zip(wishes, pair))
        if len(pair) == 2 and spoken and len(" ".join(pair).split()) <= 16 and not self.mistaken_pair(pair, purposes):
            wished = Text.lowered(" ".join(pair))
            self.order.set_aside = [r for r in self.order.requirements if Text.lowered(r["value"]) in wished]
            self.order.requirements = [r for r in self.order.requirements if Text.lowered(r["value"]) not in wished]
            self.order.pair_fallback = PLAIN_QUESTION
            if self.order.brand and set(Text.word_list(self.order.brand)) <= set(Text.word_list(wished)):
                self.order.pair_brand, self.order.brand = self.order.brand, ""
            self.order.pair_wishes = list(pair)
            if not self.order.pair_brand:
                maker = next((w for w in pair if Text.word_list(w)[-1:] == ["brand"] and len(w.split()) >= 2), "")
                self.order.pair_brand = " ".join(maker.split()[:-1]).strip(" ,.")
            self.order.pair_question = f"Which matters more to you: {pair[0]} or {pair[1]}?"
            return self.order.pair_question
        if self.order.invited or not self.interviewer.to_ask:
            return PLAIN_QUESTION
        return ""
    def beside_a_purpose(self, request: str, requirement: Dict[str, Any], uses: List[str]) -> str:
        words = [w for w in Text.word_list(Text.lowered(requirement["value"])) if len(w) >= 4]
        if not words or re.search(r"\d", requirement["value"]) or not uses:
            return ""
        named = {Text.singular(w) for w in Text.word_list(self.order.item) if len(w) >= 3}
        found = ""
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("’", "'").split())):
            said = Text.word_list(sentence)
            if not all(w in said for w in words):
                continue
            if (re.search(r"\d", sentence) or {Text.singular(w) for w in said} & named
                    or not any(len(Text.content_words(use) & Text.content_words(sentence)) >= 2 for use in uses)):
                return ""
            for clause in re.split(r",\s*(?:and|but|while)\s+|;\s*", sentence):
                if not all(w in Text.word_list(clause) for w in words):
                    continue
                tokens = clause.strip(" .!").split()
                start = next((i for i, w in enumerate(tokens) if Text.lowered(w).endswith("ing") and len(w) > 5), None)
                position = next((i for i, w in enumerate(tokens) if words[0] in Text.lowered(w)), None)
                if start is None or position is None or start >= position:
                    return ""
                found = " ".join(tokens[start:])
        return found
    def unread_purposes(self, request: str, purposes: List[str], pair: List[str]) -> List[str]:
        known = [Text.content_words(p) for p in purposes]
        taken = [Text.content_words(r["value"]) for r in self.order.requirements] + [Text.content_words(w) for w in pair]
        priority = Text.content_words(self.order.priority)
        named = Text.content_words(self.order.item) | Text.content_words(self.order.category) | {Text.singular(w) for w in Text.word_list(self.order.item) if len(w) >= 3}
        out: List[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(str(request).replace("’", "'").split())):
            lead = re.match(r"(?:since|because|as)\s+(?:i'm|i am|we're|we are)\s+([^,]{8,80}),", Text.lowered(sentence))
            if lead and Text.lowered(lead.group(1)).split()[0].endswith("ing"):
                found = lead.group(1).strip()
                words = Text.content_words(found)
                if words and not words & named and not any(len(k & words) >= 2 for k in known) and not any(t and t <= words for t in taken):
                    out.append(found)
                    known.append(words)
            if re.search(r"\d", sentence) or sentence.rstrip().endswith("?"):
                continue
            for clause in re.split(r",\s*(?:and|but|while)\s+|;\s*", sentence):
                said = Text.content_words(clause)
                if (NOT_A_PROPERTY.search(Text.lowered(clause)) or any(len(k & said) >= 2 or (k and k <= said) for k in known)
                        or any(t and t <= said for t in taken) or len(priority & said) >= 2
                        or re.search(r"\b(?:ask|check|confirm|justify|order|choose|pick|select|find)\b", Text.lowered(clause))):
                    continue
                for link in re.finditer(r"\b(?:for|match(?:es|ing)?|follow(?:s|ing)?|correspond(?:s|ing)?|based on|goal is|purpose is|use is|meant|intended)\b", Text.lowered(clause)):
                    words = clause[link.end():].strip(" .!").split()
                    start = next((i for i, w in enumerate(words) if Text.lowered(w) in ("a", "an", "the", "my", "our")
                                  or (Text.lowered(w).endswith("ing") and len(w) > 5)), None)
                    if start is None or len(words) - start < 3:
                        continue
                    found = " ".join(words[start:])
                    if Text.content_words(found) & named:
                        continue
                    if not any(self.within(found, o) for o in out):
                        out.append(found)
                        known.append(Text.content_words(found))
                    break
                else:
                    doing = re.match(r"\s*(?:i'm|i am|we're|we are)\s+(?=[a-z]{3,}ing\b)", clause, re.I)
                    found = clause[doing.end():].strip(" .!") if doing else ""
                    if len(found.split()) >= 4 and not Text.content_words(found) & named and not any(self.within(found, o) for o in out):
                        out.append(found)
                        known.append(Text.content_words(found))
        return out
    def grounded_priority(self, priority: Any, source: str) -> str:
        text = " ".join(str(priority or "").split())
        words, have = Text.content_words(text), Text.content_words(source)
        if not words:
            return ""
        if re.search(r"\b(cheap|price|cost|expensive|afford)", Text.lowered(text)):
            return text if words <= have else ""
        return text if words & have else ""
    def kept_under(self, requirement: Dict[str, Any], adds: List[Any]) -> bool:
        pages = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        for add in adds:
            if not isinstance(add, dict) or not str(add.get("property") or "").strip() or Text.lowered(add.get("property")) == Text.lowered(requirement["property"]):
                continue
            probe = {"property": str(add["property"]), "value": requirement["value"], "written": [], "alike": [], "filed": True}
            related = self.matcher.own_labels(probe)[0]
            for page in pages:
                for label, value in self.matcher.named_facts(page, probe):
                    if label in related and any(self.matcher.entry_says(entry, probe, label) for entry in Text.entries(value)):
                        return True
        return False
    @staticmethod
    def same_value(one: Any, other: Any) -> bool:
        def form(text: Any) -> str:
            return " ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(text)))
        return form(one) == form(other)
    def mentions(self, line: str, requirement: Dict[str, Any]) -> bool:
        said = Text.stems(line) | Text.name_words(line)
        named = Text.name_words(requirement["property"])
        if named and named <= said:
            return True
        return any(set(form.split()) <= said for form in self.matcher.forms(requirement) if form)
    def within(self, inner: str, outer: str) -> bool:
        words = Text.content_words(inner)
        return bool(words) and words <= Text.content_words(outer)
    def mistaken_pair(self, pair: List[str], purposes: List[str]) -> bool:
        said = [Text.content_words(p) for p in purposes]
        priority = Text.content_words(self.order.priority)
        named = Text.content_words(self.order.item) | Text.content_words(self.order.category)
        if all(Text.content_words(w) and Text.content_words(w) <= named for w in pair):
            return True
        return any(any(Text.content_words(w) & s for s in said) or len(Text.content_words(w) & priority) >= 2 for w in pair)
    def own_name(self, add: Any) -> bool:
        if not isinstance(add, dict) or re.search(r"\d", str(add.get("value") or "")):
            return False
        value = {Text.singular(w) for w in Text.word_list(Text.lowered(add.get("value")))}
        name = {Text.singular(w) for w in Text.word_list(Text.lowered(add.get("property")))}
        if not re.search(r"[&,/+]|\band\b", Text.lowered(self.order.item)):
            name |= {Text.singular(w) for w in Text.word_list(Text.lowered(self.order.item))}
        return bool(value) and value <= name
    def hear(self) -> Tuple[bool, bool]:
        changed = asks = False
        while self.catalog.heard:
            line = self.catalog.heard.pop(0)
            parts = [part.strip() for part in line.split("\n") if part.strip()]
            if len(parts) > 1:
                self.catalog.heard[:0] = parts
                continue
            before = [(r["property"], r["value"]) for r in self.order.requirements]
            held, changed_so_far = list(self.order.requirements), changed
            limit_before = self.order.budget
            said = f"{self.order.state_text()}\n\n"
            if self.interviewer.awaiting_answer and self.interviewer.last_question:
                said += f"The assistant had just asked: {self.interviewer.last_question}\n"
            said += f"The shopper's new line:\n{line}"
            if self.interviewer.awaiting_answer and self.restates_budget(line):
                heard = self.desk.listed_fields(None)
                if line != self.order.shelf_said:
                    self.order.shelf_said, self.catalog.shelf_cache = line, (-1, "")
            else:
                heard = self.desk.listed_fields(self.desk.ask(SEAT_LINE, said))
                if (self.interviewer.awaiting_answer and self.interviewer.pair_open and self.order.pair_wishes
                        and not any(isinstance(a, dict) and self.order.is_wish(str(a.get("value") or "")) for a in heard.get("adds") or [])
                        and sum(1 for wish in self.order.pair_wishes if Text.content_words(wish) & Text.content_words(line)) == 1):
                    again = self.desk.listed_fields(self.desk.ask(SEAT_LINE, said, second=True))
                    if any(isinstance(a, dict) and self.order.is_wish(str(a.get("value") or "")) for a in again.get("adds") or []):
                        heard = again
                    if not any(isinstance(a, dict) and self.order.is_wish(str(a.get("value") or "")) for a in heard.get("adds") or []):
                        chosen = next(wish for wish in self.order.pair_wishes if Text.content_words(wish) & Text.content_words(line))
                        heard["adds"] = [a for a in heard.get("adds") or [] if isinstance(a, dict)] + [self.wish_add(chosen)]
                swap = re.search(r"\binstead of\b", Text.lowered(line))
                def real_adds(reading: Dict[str, Any]) -> List[Dict[str, Any]]:
                    kept = []
                    for a in reading.get("adds") or []:
                        value = Text.lowered(a.get("value")) if isinstance(a, dict) else ""
                        if not isinstance(a, dict):
                            continue
                        if re.fullmatch(SAID_NEEDED, value):
                            continue
                        named = {w for n in self.interviewer.asked[-4:] for w in Text.name_words(n)}
                        if self.interviewer.awaiting_answer and value and (Text.name_words(value) <= named or Text.name_words(value) == Text.name_words(self.order.item)):
                            continue
                        if swap and value and value in Text.lowered(line)[swap.end():] and value not in Text.lowered(line)[:swap.start()]:
                            continue
                        kept.append(a)
                    return kept
                found = [a for a in heard.get("adds") or [] if isinstance(a, dict)]
                if (len(real_adds(heard)) < len(found) or (self.interviewer.awaiting_answer and not found)
                        or any(self.own_name(a) for a in found)
                        or (not found and heard.get("drops") and TAKES_BACK.search(Text.lowered(line)))):
                    again = self.desk.listed_fields(self.desk.ask(SEAT_LINE, said, second=True))
                    if real_adds(again):
                        heard = again
                if any(isinstance(a, dict) and re.fullmatch(SAID_NEEDED, Text.lowered(a.get("value"))) for a in heard.get("adds") or []):
                    mended = []
                    for a in heard["adds"]:
                        words = str(a.get("property") or "").split() if isinstance(a, dict) else []
                        if words and re.fullmatch(SAID_NEEDED, Text.lowered(a.get("value"))) and len(words) >= 2 and Text.says_value(line, words[-1]):
                            a = dict(a, property=" ".join(words[:-1]), value=words[-1])
                        mended.append(a)
                    heard["adds"] = mended
                if heard.get("adds") and not real_adds(heard) and self.interviewer.awaiting_answer:
                    said_so = re.search(r"\bthe ([a-z0-9 \-]+?) (?:is|are) (?:required|needed|necessary|essential|a must)\b", Text.lowered(line))
                    named = Text.name_words(self.order.item) | {w for n in self.interviewer.asked[-4:] for w in Text.name_words(n)}
                    left = [w for w in Text.word_list(said_so.group(1)) if Text.singular(w) not in named and w not in ("of", "a", "an")] if said_so else []
                    asked_for = next((n for n in reversed(self.interviewer.asked[-4:]) if Text.name_words(n) & Text.name_words(said_so.group(1))), "") if said_so else ""
                    if left and len(left) <= 3 and asked_for:
                        heard["adds"] = [{"property": asked_for, "value": " ".join(left)}]
                if heard.get("adds"):
                    heard["adds"] = real_adds(heard)
            answering, self.interviewer.awaiting_answer = self.interviewer.awaiting_answer, False
            nothing = answering and not any(heard.get(k) for k in ("adds", "drops", "budget", "priority", "about_change"))
            if nothing and len(self.interviewer.current_names) == 2:
                heard["asks"] = False
            given = replaced = 0
            withdrawn = []
            held_before = list(self.order.requirements)
            if heard.get("about_change") is True:
                heard["budget"] = 0
            beside = {Text.lowered(a.get("property")) for a in heard.get("adds") or [] if isinstance(a, dict) and a.get("keeps_earlier") is True}
            drops = [] if answering and not TAKES_BACK.search(Text.lowered(line)) else (heard.get("drops") or [])
            for name in [n for n in drops if Text.lowered(n) not in beside]:
                named, _, said_value = str(name).partition("=") if "=" in str(name) else str(name).partition(":")
                named, said_value = Text.lowered(named).strip(), Text.lowered(said_value).strip()
                kept = [r for r in self.order.requirements if Text.lowered(r["property"]) != named
                        or (said_value and not self.same_value(r["value"], said_value) and Text.closed_up(r["value"]) != Text.closed_up(said_value))
                        or not (self.mentions(line, r) or (TAKES_BACK.search(Text.lowered(line)) and self.kept_under(r, heard.get("adds") or [])))]
                changed = changed or len(kept) != len(self.order.requirements)
                replaced += len(self.order.requirements) - len(kept)
                withdrawn += [r for r in self.order.requirements if r not in kept]
                self.order.requirements = kept
            added = [Text.lowered(a.get("value")) for a in heard.get("adds") or [] if isinstance(a, dict) and str(a.get("value") or "").strip()
                     and not NOT_A_PROPERTY.search(Text.lowered(f"{a.get('property')} {a.get('value')}"))]
            if TAKES_BACK.search(Text.lowered(line)) and not withdrawn:
                new_names = [a for a in heard.get("adds") or [] if isinstance(a, dict) and a.get("keeps_earlier") is not True
                             and not any(Text.name_words(a.get("property")) == Text.name_words(r["property"]) for r in self.order.requirements)]
                filed = [r for r in self.order.requirements if new_names and self.kept_under(r, new_names)]
                if len(filed) == 1:
                    self.order.requirements = [r for r in self.order.requirements if r is not filed[0]]
                    withdrawn += filed
                    replaced += 1
                    changed = True
            if heard.get("drops") and added and not withdrawn:
                said = Text.stems(line)
                gone = [r for r in self.order.requirements if Text.lowered(r["value"]) not in added and Text.stems(r["value"])
                        and Text.stems(r["value"]) <= said and Text.says_value(line, r["value"])
                        and not re.search(r"\bstill\b[^.!?]*" + re.escape(Text.lowered(r["value"])), Text.lowered(line))]
                if len(gone) == 1:
                    self.order.requirements = [r for r in self.order.requirements if r is not gone[0]]
                    withdrawn += gone
                    replaced += 1
                    changed = True
            if withdrawn and not added:
                for old in withdrawn:
                    was = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", Text.lowered(old["value"]).strip())
                    if not was:
                        continue
                    others = [m for m in re.finditer(r"(?<![a-z0-9.,])(\d+(?:\.\d+)?)\s?([a-z]+|\")(?![a-z0-9])", Text.lowered(line))
                              if Text.same_unit(Text.unit_word(m.group(2)), Text.unit_word(was.group(2))) and float(m.group(1)) != float(was.group(1))]
                    if len({float(m.group(1)) for m in others}) == 1:
                        heard["adds"] = list(heard.get("adds") or []) + [{"property": old["property"], "value": f"{others[0].group(1)} {was.group(2)}"}]
            for add in heard.get("adds") or []:
                if not isinstance(add, dict) or not str(add.get("value") or "").strip():
                    continue
                new = self.order.requirement(add)
                p_words, v_words = Text.word_list(Text.lowered(new["property"])), Text.word_list(Text.lowered(new["value"]))
                if (len(p_words) >= 2 and len(v_words) >= 2 and p_words[-1] == v_words[-1] in ("type", "kind") and not re.search(r"\d", new["value"])
                        and not set(p_words[:-1]) & set(v_words[:-1])):
                    new["property"], new["value"] = new["value"], " ".join(p_words[:-1])
                elif (len(p_words) >= 2 and p_words[-1] == "type" and not re.search(r"\d", new["value"])
                        and re.search(r"\b" + re.escape(Text.lowered(new["value"])) + r"\s+type\b", Text.lowered(line))):
                    new["property"], new["value"] = f"{Text.lowered(new['value'])} type", " ".join(p_words[:-1])
                if self.own_name(new):
                    prop = r"\s+".join(re.escape(w) for w in Text.word_list(Text.lowered(new["property"])))
                    of = re.search(r"\bthe ([a-z][a-z \-]{1,40}?) (?:must|has to|have to|should|needs? to) be (?:of |made of )?(?:the |a |an )?(?:[a-z\-]+ ){0,3}?" + prop + r"\b", Text.lowered(line))
                    thing = [w for w in Text.word_list(of.group(1)) if Text.singular(w) not in Text.name_words(self.order.item)] if of else []
                    if not thing or len(thing) > 3:
                        continue
                    new["value"] = " ".join(thing)
                if (not re.search(r"colou?r", Text.lowered(new["property"])) and not re.search(r"\d", new["value"])
                        and not re.search(r"\b(?:type|kind)\b", Text.lowered(new["property"]))):
                    own, tokens = Text.name_words(self.order.item), Text.lowered(new["value"]).split()
                    rest = [w for i, w in enumerate(tokens)
                            if not (("-" in w and Text.singular(w.split("-")[0]) in own) or (i == len(tokens) - 1 and i > 0 and Text.singular(w) in own))]
                    if rest and len(rest) < len(tokens):
                        new["value"] = " ".join(rest)
                if NOT_A_PROPERTY.search(Text.lowered(f"{new['property']} {new['value']}")):
                    continue
                if not Text.says_value(line, new["value"]):
                    continue
                if not re.search(r"\d", new["value"]) and any(self.within(new["value"], purpose) for purpose in self.order.purposes):
                    continue
                stands_now = next((now for now, before in self.order.took_back.items() if Text.lowered(before) == Text.lowered(new["value"])), None)
                if stands_now is not None and not Text.says_value(line, stands_now):
                    for held_one in held_before:
                        if (Text.lowered(held_one["property"]) == Text.lowered(new["property"]) or Text.lowered(held_one["value"]) == Text.lowered(stands_now)) \
                                and held_one not in self.order.requirements:
                            self.order.requirements.append(held_one)
                            withdrawn = [w for w in withdrawn if w is not held_one]
                            replaced = max(0, replaced - 1)
                    continue
                same = next((r for r in self.order.requirements if Text.lowered(r["property"]) == Text.lowered(new["property"])), None)
                if same is not None and self.same_value(same["value"], new["value"]):
                    continue
                if same is None and any(self.same_value(r["value"], new["value"]) for r in self.order.requirements):
                    continue
                if same is not None and answering and not self.mentions(line, same):
                    new["property"], same = "requirement", None
                if same is not None and add.get("keeps_earlier") is True and not answering:
                    new["also"], same = same["value"], None
                if same is not None and answering and not TAKES_BACK.search(Text.lowered(line)):
                    old_words = set(Text.word_list(Text.lowered(same["value"])))
                    new_words = set(Text.word_list(Text.lowered(new["value"])))
                    if old_words and new_words and (old_words < new_words or new_words < old_words):
                        if new_words < old_words:
                            continue
                        self.order.requirements.remove(same)
                        self.order.opening = [r for r in self.order.opening if Text.lowered(r["value"]) != Text.lowered(same["value"])]
                    else:
                        new["also"] = same["value"]
                    same = None
                if same is not None:
                    self.order.requirements.remove(same)
                    replaced += 1
                    withdrawn.append(same)
                else:
                    given += 1
                old = {Text.closed_up(w) for r in withdrawn for w in [r["value"]] + list(r.get("written") or []) + list(r.get("alike") or [])}
                for kind in ("written", "alike"):
                    if isinstance(new.get(kind), list):
                        new[kind] = [w for w in new[kind] if Text.closed_up(str(w)) not in old]
                if answering and self.interviewer.pair_open and self.order.is_wish(new["value"]):
                    new["chosen"] = True
                instead_of = [r for r in withdrawn if Text.lowered(r["property"]) == Text.lowered(new["property"])] or (withdrawn if len(withdrawn) == 1 else [])
                unit_before = re.fullmatch(r"\d+(?:\.\d+)?\s*([a-z][a-z ]*)", Text.lowered(instead_of[0]["value"]).strip()) if instead_of else None
                if unit_before and re.fullmatch(r"\d+(?:\.\d+)?", new["value"].strip()):
                    new["value"] = f"{new['value'].strip()} {unit_before.group(1).strip()}"
                    new["filed"] = False
                if instead_of and Text.lowered(instead_of[0]["value"]) != Text.lowered(new["value"]):
                    self.order.took_back[Text.lowered(new["value"])] = instead_of[0]["value"]
                    self.matcher.read_cache.clear()
                self.order.requirements.append(new)
                changed = True
                if "brand" in Text.lowered(new["property"]) and new.get("chosen"):
                    self.order.brand, self.order.brand_chosen = self.order.pair_brand or new["value"], True
                if "brand" in Text.lowered(new["property"]):
                    self.planner.queue_filters()
                    self.planner.add_queries([f"{new['value']} {self.order.item}", new["value"]], first=True)
            if answering and self.interviewer.pair_open and self.order.set_aside and any(
                    r not in self.order.set_aside and not self.order.is_wish(r["value"]) and Text.says_value(line, r["value"]) for r in held):
                back = [r for r in self.order.set_aside if not any(self.same_value(r["value"], x["value"]) for x in self.order.requirements)]
                self.order.requirements = [r for r in self.order.requirements if r in held] + back + [r for r in self.order.requirements if r not in held]
                self.order.opening += [dict(r) for r in back if r.get("stated")]
                self.order.set_aside = []
                changed = changed or bool(back)
            if sorted((Text.lowered(p), Text.lowered(v)) for p, v in before) == sorted((Text.lowered(r["property"]), Text.lowered(r["value"])) for r in self.order.requirements):
                self.order.requirements, changed, given, replaced = held, changed_so_far, 0, 0
            else:
                self.order.fresh[line] = [r["value"] for r in self.order.requirements if r not in held]
            budget = Text.money(heard.get("budget"))
            if budget is not None and not Text.says_amount(line, budget):
                budget = None
            if budget is not None and any(abs(budget - float((self.catalog.stock.get(k) or {}).get("price") or -1.0)) < 0.005 for k in self.catalog.cart):
                budget = None
            if budget is not None and budget != self.order.budget:
                lowered_limit = self.order.budget is None or budget < self.order.budget
                self.order.budget, changed = budget, True
                self.order.fresh_limit.add(line)
                if lowered_limit:
                    self.planner.queue_filters()
            if self.grounded_priority(heard.get("priority"), line) and not (answering and self.interviewer.pair_open):
                self.order.priority, changed = self.grounded_priority(heard.get("priority"), line), True
                self.sieve.figures, self.sieve.figure_unit = {}, ""
            moved = [(r["property"], r["value"]) for r in self.order.requirements] != before or self.order.budget != limit_before
            wants_reply = heard.get("asks") is True and heard.get("about_change") is not True
            asked_back = heard.get("asks") is True and not any(heard.get(k) for k in ("adds", "drops", "budget", "priority"))
            spoke_of_else = replaced > 0 or self.order.budget != limit_before or heard.get("about_change") is True or asked_back
            if (replaced > 0 or self.order.budget != limit_before) and not self.catalog.cart:
                self.order.changed_early = True
            if replaced > 0 or self.order.budget != limit_before or (moved and not answering):
                self.order.moved_on = True
            was_answer = answering and not spoke_of_else
            if (wants_reply or (moved and not was_answer)) and line not in self.order.unanswered:
                self.order.unanswered.append(line)
            if answering:
                self.interviewer.answered(line, given, spoke_of_else=spoke_of_else, about=" ".join(Text.word_list(Text.lowered(heard.get("about")))), nothing=nothing)
            self.order.dropped |= {Text.closed_up(r["value"]) for r in withdrawn if r not in self.order.requirements}
            if changed:
                self.order.changed_at = self.session.turn
                gone_words = [set(Text.word_list(Text.closed_up(r["value"]))) for r in withdrawn if r not in self.order.requirements]
                kept_words = {w for v in self.planner.values() for w in Text.word_list(v)}
                gone_words = [words - kept_words for words in gone_words if words - kept_words]
                if gone_words:
                    self.planner.queries = [q for q in self.planner.queries if not any(words <= set(q.split()) for words in gone_words)]
                if given or replaced:
                    self.planner.fresh_rounds = 2
                    self.planner.shelf_words_used = 0
                    self.planner.planned = []
                self.planner.add_queries(self.planner.value_queries() + list(heard.get("queries") or []), first=True)
                self.planner.search_room = max(self.planner.search_room, len(self.planner.sent) + 16)
                self.planner.search_asks = 0
                self.planner.open_room = max(self.planner.open_room, len(self.catalog.pages) + 40)
                self.sieve.figure_room = max(self.sieve.figure_room, len(self.sieve.figures) + 3 * JUDGE_CARDS)
                self.judge.judge_room = max(self.judge.judge_room, self.judge.judged_count + 36)
            if heard.get("asks") and not heard.get("about_change"):
                asks = True
                self.pending_line = line
            if heard.get("about_change") is True and self.reply_sent and self.reply_sent[0] == self.session.turn and self.replies_again < 2:
                self.replies_again += 1
                for said_line in self.reply_sent[1].split("\n"):
                    text = said_line[2:] if said_line.startswith("- ") else said_line
                    if text.strip() and text not in self.order.unanswered:
                        self.order.unanswered.append(text)
                self.reply_sent = None
        return changed, asks
    def restates_budget(self, line: str) -> bool:
        if self.order.budget is None or f"{self.order.budget:.2f}" not in line.replace(",", ""):
            return False
        figures = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", line.replace(",", ""))]
        if any(abs(figure - self.order.budget) > 0.005 for figure in figures):
            return False
        return not any(self.mentions(line, r) for r in self.order.requirements + self.order.set_aside)
    def hear_quietly(self) -> None:
        if self.catalog.heard:
            self.hear()


SEARCH_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one sentence: the directions these searches take that the searches already sent have not."},
    "queries": Schema.strings("Ten new searches, each from a different direction, of one to six words."),
}, ["reason", "queries"])

SEARCH_INSTRUCTIONS = """You are the search planner of a shopping assistant. The assistant is looking for the store listing that meets every requirement of the shopper, and you write its next catalogue searches.

WHAT YOU ARE GIVEN
The request, the requirements and the budget, the searches already sent, and the titles of listings read so far.

HOW THE STORE'S SEARCH WORKS
It ranks listings by how many words of the search their text carries, the title counting most, and shows ten listings for a search. A listing is reached only by a search whose words it carries: the words of the searches decide which listings can be found at all, and searches that share most of their words bring back the same listings.

WHAT YOU RETURN
Ten searches that go in different directions - not ten rewordings of one search - and a reason first.

THE DIRECTIONS - cover as many as fit the request:
1. The kind of item as sellers name it in titles: the words the titles seen use for what each article is, other names for the same kind of item, a wider name and a narrower one.
2. One required value with a name for the item, for each value in turn; then two values together. Never stack every value into one search.
3. A value as sellers write it in other ways: another word with the same meaning, another spelling, an amount written joined to its unit and apart from it, the same amount in another unit.
4. The required values without any name for the item: listings that state them whatever they call the item.
5. The shopper's use, or the person the item is for, in the words a seller would put in a title.
6. The main title words of a listing seen that came close, with the value it did not state.
7. A value the shopper has just changed or added, or a lower limit: the new value first, with the item's names, in several of the forms above.
8. The store's own words for a value, when the request shows them: those words with the item's names, and with the other values.
9. The item's own names alone, or with one word the titles seen use often, and no value at all: a listing that states a value only in its variants or its facts carries none in its title, and a search with the value ranks it below listings that do.
10. A colour, a pattern or a size is most often named only in a variant: search the item with the other values, then read the variants.

RULES
- Mix short searches of one or two words with longer ones of four to six words.
- Never send a search again in another order or with a filler word added: it brings back the same listings.
- When the searches already sent share most of their words, go in the directions they have not tried.
- Write an amount as listings write it, joined to its unit, in one search; apart from it in another.
- reason: written first, in one sentence: the directions these searches take that the searches already sent have not."""

SEAT_SEARCH = ModelSeat('search_plan', 'New catalogue searches.', SEARCH_INSTRUCTIONS, SEARCH_SCHEMA, 1024, (GLM, GLM, LUNA))

BEST_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one sentence: which better figures the searches aim at and how listings write them."},
    "queries": Schema.strings("Up to eight new searches of two to six words each, different from those tried."),
}, ["reason", "queries"])

BEST_INSTRUCTIONS = """You write catalogue searches for a shopping assistant. A shopper wants, among the store listings that meet every requirement, the one that stands best by a priority of theirs (the most of something, the largest, the longest, ...). The assistant has found some fitting listings; you write searches that could reach fitting listings that stand better still.

WHAT YOU ARE GIVEN
The request, the requirements, the priority, the best figures found so far with the words of the listings that carry them, and the searches already tried.

HOW TO WRITE THEM
1. Listings state such a figure in their title or variant name. A search that carries a figure written the way a listing writes it brings up the listings that state it - and sellers write it both joined to its unit and apart from it, which the store treats as different words. So write searches made of the item words, the required values and a figure for the priority that is better than the best found so far: several steps beyond it, one figure per search, some written joined and some apart.
2. A title that lists several amounts lists the variants sold under it: when such a title shows an amount better than the best found, search that title's main words with that amount and the required values.
3. Add one search with the words sellers use for the better end of that priority.
4. Keep every search short; never repeat a search already tried.

reason: written first, in one sentence: which better figures the searches aim at and how listings write them."""

SEAT_BEST = ModelSeat('search_better', "Catalogue searches for listings that stand better by the shopper's priority.", BEST_INSTRUCTIONS, BEST_SCHEMA, 1024, (GLM, GLM, LUNA))

WORDS_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one sentence: what the value means and how sellers write it."},
    "words": {"type": "array", "items": {"type": "string"}, "description": "Up to six, the most usual first; empty when there is none."}}, ["reason", "words"])

WORDS_INSTRUCTIONS = """You help a shopping assistant search a store. A shopper wants a certain value for a property of an item. List the other words or spellings that sellers use in listing titles and variant names for that same value.
- Take: synonyms and other spellings of a word; the same amount in other units and notations.
- Leave out: a part of it, a shade of it, a wider or a narrower kind of it. Only words that mean the same value.
- Up to six, the most usual first; empty when there is none.
reason: written first, in one sentence: what the value means and how sellers write it."""

SEAT_WORDS = ModelSeat('words', 'Other words sellers use for a value.', WORDS_INSTRUCTIONS, WORDS_SCHEMA, 1024, (GLM, GLM, LUNA))


class SearchPlanner(Part):
    def __init__(self) -> None:
        self.queries: List[str] = []
        self.sent: Set[str] = set()
        self.searched: List[str] = []
        self.shelf_words_used = 0
        self.search_room = SEARCH_LIMIT
        self.open_room = OPEN_LIMIT
        self.widened = False
        self.probed: Set[str] = set()
        self.kin_done: Set[str] = set()
        self.pending_filters: List[Dict[str, Any]] = []
        self.planned: List[str] = []
        self.planned_at = -1
        self.search_asks = 0
        self.kin_swept: Set[str] = set()
        self.fresh_rounds = 0
        self.near_done: Set[str] = set()
        self.printed_done: Set[str] = set()
    def add_queries(self, queries: List[Any], first: bool = False) -> None:
        fresh = []
        for query in queries:
            text = Text.lowered(" ".join(str(query or "").split())[:QUERY_LENGTH])
            if text and text not in self.sent and text not in self.queries and text not in fresh:
                fresh.append(text)
        self.queries = fresh + self.queries if first else self.queries + fresh
    def values(self) -> List[str]:
        out = []
        for requirement in self.order.requirements:
            named = Text.name_words(requirement["property"])
            words = Text.word_list(Text.closed_up(requirement["value"]))
            if re.search(r"\d[.,]\d", requirement["value"]):
                words = [w.strip(".,;:()") for w in Text.closed_up(requirement["value"]).split() if w.strip(".,;:()")]
            kept = [w for w in words if Text.singular(w) not in named]
            if re.search(r"\d", requirement["value"]) and not re.search(r"\d", " ".join(kept)):
                kept = words
            if kept and not re.search(r"\d", " ".join(kept)) and all(len(w) <= 3 for w in kept):
                kept = []
            out.append(" ".join(kept or words))
        return out
    def item_words(self) -> List[str]:
        return [w for w in Text.word_list(self.order.category or self.order.item) if len(w) >= 3 and w not in ("and", "the", "for", "with", "other")]
    def value_queries(self) -> List[str]:
        values, item = self.values(), self.order.category or self.order.item
        label = self.store_label()
        if label and values and Text.lowered(label) != Text.lowered(item):
            self.add_queries([f"{' '.join(values)} {label}", f"{values[-1]} {label}"])
        if not values or not item:
            return [item] if item else []
        every, newest = " ".join(values), values[-1]
        out = [f"{every} {item}", f"{newest} {item}", f"{values[0]} {item}"]
        head = (self.item_words() or [""])[-1]
        apart = re.sub(r"(\d)([a-z])", r"\1 \2", every)
        if apart != every:
            out[1:1] = [f"{apart} {item}"] + ([f"{apart} {head}"] if head and len(item.split()) > 1 else [])
            newest_apart = re.sub(r"(\d)([a-z])", r"\1 \2", newest)
            if newest_apart != newest and len(values) > 1:
                out.append(f"{newest_apart} {item}")
        for requirement in self.order.requirements:
            for other in self.spellings(requirement)[1:4]:
                if Text.closed_up(other) != Text.closed_up(requirement["value"]):
                    turned = " ".join(Text.closed_up(other if r is requirement else r["value"]) for r in self.order.requirements)
                    out.append(f"{turned} {item}")
        out += [f"{every} {w}" for w in self.item_words()[:3]]
        if len(values) >= 2:
            out.append(every)
        out += [f"{newest} {w}" for w in self.item_words()[:2]]
        earlier = " ".join(values[:-1])
        whole = " ".join(Text.word_list(Text.closed_up(self.order.requirements[-1]["value"])))
        if whole and not re.search(r"\d", whole):
            head, _, tail = whole.rpartition(" ")
            turned = tail[:-1] if tail.endswith("s") and len(tail) > 3 else tail + "s"
            forms = ([whole] if whole != newest else []) + [f"{head} {turned}".strip()]
            out[3:3] = [" ".join(f"{earlier} {form} {item}".split()) for form in forms]
        for requirement in reversed(self.order.requirements):
            rest = " ".join(Text.closed_up(r["value"]) for r in self.order.requirements if r is not requirement)
            for other in (requirement.get("alike") or [])[:2]:
                if Text.closed_up(other) != Text.closed_up(requirement["value"]):
                    out.append(" ".join(f"{rest} {Text.closed_up(other)} {item}".split()))
        if len(values) >= 2:
            out += [f"{Text.closed_up(v)} {item}" for v in values]
            if len(values) >= 3:
                out += [" ".join(Text.closed_up(v) for j, v in enumerate(values) if j != i) + f" {item}" for i in range(len(values))]
        return out
    @staticmethod
    def spellings(requirement: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for form in [requirement["value"]] + list(requirement.get("written") or []):
            if not Text.shortened(form, requirement["value"]) and Text.closed_up(form) not in [Text.closed_up(f) for f in out]:
                out.append(form)
        return out
    def new_amount_queries(self) -> List[str]:
        newest = list(self.order.fresh.values())[-1] if self.order.fresh else []
        requirement = next((r for r in self.order.requirements if r["value"] in newest and re.search(r"\d", str(r["value"]))), None)
        if requirement is None:
            return []
        forms = [Text.closed_up(f) if i == 0 else " ".join(Text.lowered(f).split()) for i, f in enumerate(self.spellings(requirement)) if re.search(r"\d", f)][:3]
        limit = self.order.budget
        titles = [row.get("title") for k, row in self.catalog.rows.items() if k in self.catalog.shelf
                  and (limit is None or not isinstance(row.get("price"), (int, float)) or row["price"] <= limit)]
        titles += [p.get("title") for p in self.catalog.pages.values() if self.catalog.in_category(p)]
        known = set(self.item_words()) | {w for f in forms for w in Text.word_list(f)}
        counts: Dict[str, int] = {}
        for title in titles:
            for word in set(Text.word_list(title)):
                if len(word) >= 4 and not re.search(r"\d", word) and word not in known:
                    counts[word] = counts.get(word, 0) + 1
        words = [w for w, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= 2][:4]
        return [f"{w} {forms[0]}" for w in words] + [f"{w} {f}" for f in forms[1:3] for w in words[:2]]
    def shelf_queries(self, count: int) -> List[str]:
        titles = [self.catalog.rows[k].get("title") for k in self.catalog.shelf if k in self.catalog.rows]
        titles += [p.get("title") for p in self.catalog.pages.values() if self.catalog.in_category(p)]
        if len(titles) < 3 or not self.order.requirements:
            return []
        known = set(self.item_words()) | {w for v in self.values() for w in Text.word_list(Text.closed_up(v))}
        counts: Dict[str, int] = {}
        first_seen: Dict[str, int] = {}
        for title in titles:
            for word in set(Text.word_list(title)):
                if len(word) >= 4 and not word.isdigit() and word not in known:
                    counts[word] = counts.get(word, 0) + 1
                    first_seen.setdefault(word, len(first_seen))
        least = max(2, len(titles) // 12)
        common = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]])) if n >= least]
        chosen = common[self.shelf_words_used: self.shelf_words_used + count]
        self.shelf_words_used += len(chosen)
        every, newest = " ".join(self.values()), self.values()[-1]
        head = (self.item_words() or [""])[-1]
        out = ([f"{w} {every}" for w in chosen] + [f"{w} {every} {head}" for w in chosen[: count // 2]]
               + [f"{w} {newest}" for w in chosen[: count // 2]])
        spelt = " ".join(Text.closed_up(next((w for w in r.get("written") or [] if not Text.shortened(w, r["value"]) and not re.search(r"\d", w)
                                         and Text.closed_up(w) != Text.closed_up(r["value"])), r["value"])) for r in self.order.requirements)
        if spelt != every:
            out[1:1] = [f"{w} {spelt}" for w in chosen[:2]]
        apart = re.sub(r"(\d)([a-z])", r"\1 \2", every)
        if apart != every:
            out[1:1] = [f"{w} {apart}" for w in chosen[:3]]
        return out
    def store_forms(self, requirement: Dict[str, Any]) -> List[str]:
        texts = [str((self.catalog.rows.get(k) or {}).get("title") or "") for k in self.catalog.shelf if k in self.catalog.rows]
        for page in self.catalog.pages.values():
            if self.catalog.in_category(page):
                texts.append(str(page.get("title") or ""))
                texts += [str(v.get("options") or "") for v in page.get("variants") or [] if isinstance(v, dict)]
                texts += [str(v) for v in (page.get("facts") or {}).values()]
        value = Text.lowered(requirement["value"]).strip()
        found = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", value)
        seen: Dict[str, int] = {}
        if found:
            figure, unit = float(found.group(1)), found.group(2)
            for text in texts:
                for number, name in re.findall(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")(?![a-z0-9])", Text.lowered(text)):
                    if name == unit or not Text.same_unit(unit, name):
                        continue
                    amount = figure
                    if amount > 0 and abs(float(number) - amount) < 1e-9:
                        form = f"{amount:g}{name}" if len(name) <= 3 else f"{amount:g} {name}"
                        seen[form] = seen.get(form, 0) + 1
        else:
            words = [w for w in Text.word_list(value) if len(w) >= 4]
            for text in texts:
                for word in set(Text.word_list(text)):
                    for w in words:
                        if word != w and Text.one_letter_apart(word, w) and not word.isdigit():
                            form = " ".join(word if x == w else x for x in Text.word_list(value))
                            seen[form] = seen.get(form, 0) + 1
        return [form for form, _ in sorted(seen.items(), key=lambda kv: -kv[1])][:3]
    def printed_queries(self) -> List[str]:
        item = self.order.category or self.order.item
        head = (self.item_words() or [""])[-1]
        out: List[str] = []
        for requirement in self.order.requirements:
            rest = " ".join(Text.closed_up(r["value"]) for r in self.order.requirements if r is not requirement)
            for form in self.store_forms(requirement):
                tag = f"{Text.lowered(requirement['value'])}|{form}"
                if tag in self.printed_done:
                    continue
                self.printed_done.add(tag)
                out += [f"{rest} {form} {item}", f"{rest} {form} {head}" if head else "", f"{form} {item}", f"{rest} {form}" if rest else ""]
        return [" ".join(q.split()) for q in out if q.strip()]
    def store_queries(self) -> List[str]:
        item = self.order.category or self.order.item
        head = (self.item_words() or [""])[-1]
        out = []
        for requirement in self.order.requirements:
            rest = " ".join(Text.closed_up(r["value"]) for r in self.order.requirements if r is not requirement)
            for form in (list(requirement.get("store") or []) + list(requirement.get("sought") or []))[:4]:
                out += [f"{rest} {form} {item}", f"{form} {head}", f"{rest} {form}"]
        return [" ".join(q.split()) for q in out if q.strip()]
    def shelf_words(self, count: int, skip: int = 0) -> List[str]:
        titles = [self.catalog.rows[k].get("title") for k in self.catalog.shelf if k in self.catalog.rows]
        titles += [p.get("title") for p in self.catalog.pages.values() if self.catalog.in_category(p)]
        if len(titles) < 3:
            return []
        known = set(self.item_words()) | {w for v in self.values() for w in Text.word_list(Text.closed_up(v))}
        counts: Dict[str, int] = {}
        first_seen: Dict[str, int] = {}
        for title in titles:
            for word in set(Text.word_list(title)):
                if len(word) >= 4 and not re.search(r"\d", word) and word not in known:
                    counts[word] = counts.get(word, 0) + 1
                    first_seen.setdefault(word, len(first_seen))
        least = max(2, len(titles) // 12)
        common = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]])) if n >= least]
        return common[skip: skip + count]
    def sweep_queries(self, round_number: int) -> List[str]:
        values = [v for v in self.values() if v]
        head = (self.item_words() or [""])[-1]
        out: List[str] = []
        if round_number == 0:
            out += self.store_queries()
            if len(values) >= 2:
                out.append(" ".join(values))
            out += [f"{values[i]} {values[j]}" for i in range(len(values)) for j in range(i + 1, len(values))]
            out += [f"{v} {head}" for v in values if head and head not in v.split()]
        words = self.shelf_words(8, skip=8 * round_number)
        out += [f"{w} {head}" for w in words if head] + words[:3]
        return out
    def maker_filters(self, count: int) -> List[Dict[str, Any]]:
        label = self.store_label()
        if not label or not self.session.has("filter") or self.order.budget is None or self.order.brand:
            return []
        makers: Dict[str, int] = {}
        for page in self.catalog.pages.values():
            brand = str(page.get("brand") or "").strip()
            if brand and self.catalog.in_category(page) and Text.lowered(brand) not in NO_MAKER:
                makers[brand] = makers.get(brand, 0) + 1
        out = []
        for brand, _ in sorted(makers.items(), key=lambda kv: -kv[1]):
            tag = f"maker|{Text.lowered(label)}|{Text.lowered(brand)}"
            if tag in self.sent:
                continue
            self.sent.add(tag)
            out.append(StoreSession.call("filter", category=label, brand=brand, max_price=self.order.budget, in_stock=True, k=10))
            if len(out) >= count:
                break
        return out
    def kin_titles(self) -> List[str]:
        return list(dict.fromkeys(Text.lowered((self.catalog.pages.get(k[0]) or {}).get("title")) for k in self.judge.verdicts if self.sieve.fits_words(k)))
    def kin_pending(self) -> bool:
        return len(self.kin_done) < 6 and any(title not in self.kin_done for title in self.kin_titles())
    def kin_queries(self) -> List[str]:
        every = " ".join(self.values())
        out = []
        seeds = [k for k in self.judge.verdicts if self.sieve.fits_words(k)]
        seeds += [k for k, row in self.catalog.rows.items() if k not in self.judge.verdicts and k[0] not in self.catalog.pages and self.order.requirements
                  and all(self.matcher.says(row.get("title"), r) for r in self.order.requirements)][:2]
        for key in seeds:
            page = self.catalog.pages.get(key[0]) or self.catalog.rows.get(key) or {}
            title = Text.lowered(page.get("title"))
            if not title or title in self.kin_done or len(self.kin_done) >= 6:
                continue
            self.kin_done.add(title)
            self.friend_filters(key)
            words = Text.word_list(title)
            if len(words) < 2:
                continue
            out += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.order.item} {' '.join(words[:3])}"]
            if len(words) > 10:
                out.append(f"{every} {' '.join(words)}")
            options = " ".join(Text.word_list(self.catalog.variant_of(key).get("options")))
            if options:
                out.append(f"{options} {' '.join(words[:10])}")
            brand = Text.lowered(page.get("brand"))
            if brand and brand not in NO_MAKER:
                out.append(f"{every} {brand} {self.order.item}")
            out.append(f"{every} {' '.join(words[:2])}")
            elsewhere = {w for row in self.catalog.rows.values() if Text.lowered(row.get("title")) != title for w in Text.word_list(row.get("title"))}
            own = sorted((w for w in dict.fromkeys(words) if len(w) >= 5 and not w[0].isdigit() and w not in elsewhere), key=len, reverse=True)
            if own:
                out.append(own[0])
        return out
    def near_kin_queries(self) -> List[str]:
        fresh = {Text.lowered(v) for values in self.order.fresh.values() for v in values}
        if not fresh:
            return []
        new_words = " ".join(Text.closed_up(r["value"]) for r in self.order.requirements if Text.lowered(r["value"]) in fresh)
        out: List[str] = []
        for key, verdict in self.judge.verdicts.items():
            met = (verdict.get("met") or [])[: len(self.order.requirements)]
            if not verdict.get("kind_ok") or len(met) < len(self.order.requirements) or all(met):
                continue
            missing = [r for x, r in zip(met, self.order.requirements) if not x]
            if not all(Text.lowered(r["value"]) in fresh for r in missing):
                continue
            title = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title"))
            if not title or title in self.near_done or len(self.near_done) >= 8:
                continue
            self.near_done.add(title)
            words = Text.word_list(title)
            if len(words) >= 2:
                out += [f"{new_words} {' '.join(words[:10])}", f"{new_words} {' '.join(words[:5])}"]
        return out
    def friend_filters(self, key: Key) -> None:
        page = self.catalog.pages.get(key[0]) or {}
        label = str((page.get("category_path") or [""])[-1] or "").strip()
        if not label or not self.session.has("filter") or self.order.budget is None:
            return
        brand = str(page.get("brand") or "").strip()
        wanted = [brand] if brand and Text.lowered(brand) not in NO_MAKER else []
        if Text.label_key(label) != Text.label_key(self.store_label()):
            wanted.append("")
        for make in wanted:
            tag = f"friends|{Text.lowered(label)}|{Text.lowered(make)}"
            if tag not in self.sent and len(self.sent) < self.search_room:
                self.sent.add(tag)
                self.pending_filters.append(StoreSession.call("filter", category=label, brand=make or None, max_price=self.order.budget, in_stock=True, k=10))
    def variant_queries(self) -> List[str]:
        out = []
        for key, verdict in list(self.judge.verdicts.items()):
            if len(self.probed) >= 4:
                break
            if not verdict.get("kind_ok") or verdict.get("by") != "words" or not self.order.requirements:
                continue
            levels = [self.matcher.plain(key, r) for r in self.order.requirements]
            off = [r for r, level in zip(self.order.requirements, levels) if level < 1]
            rest = [level for r, level in zip(self.order.requirements, levels) if level >= 1]
            if not off or len(rest) != len(self.order.requirements) - len(off):
                continue
            if any(self.matcher.kind(r) not in ("colour", "amount") for r, level in zip(self.order.requirements, levels) if level < 1):
                continue
            title = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title"))
            if not title or title in self.probed:
                continue
            self.probed.add(title)
            words = " ".join(Text.word_list(title)[:10])
            for requirement in off:
                for word in [Text.closed_up(requirement["value"])] + list(requirement.get("alike") or [])[:2]:
                    out.append(f"{word} {words}")
        return out
    def label_guesses(self) -> List[str]:
        base = self.order.category or self.order.item
        if not base:
            return []
        def plural(word: str) -> str:
            if word.endswith("s"):
                return word
            if re.search(r"(x|ch|sh)$", word):
                return word + "es"
            if re.search(r"ife$", word):
                return word[:-2] + "ves"
            if re.search(r"(?:[lr]f|[oe]af)$", word):
                return word[:-1] + "ves"
            if re.search(r"[^aeiou]y$", word):
                return word[:-1] + "ies"
            return word + "s"
        out = []
        parts = base.split()
        for skip in range(min(3, len(parts))):
            phrase = " ".join(parts[skip:])
            if skip and not re.match(r"[A-Za-z]{3}", phrase):
                continue
            last = re.sub(r"([A-Za-z]+)(\W*)$", lambda m: plural(m.group(1)) + m.group(2), phrase)
            every = re.sub(r"([A-Za-z]+)(?=\s*(?:&|/|,|\band\b)|\W*$)", lambda m: plural(m.group(1)), phrase)
            out += [phrase, last, every] if not skip else [last]
        return list(dict.fromkeys(out))
    def store_label(self) -> str:
        named = self.catalog.named_shelf()
        if named:
            return named
        if self.catalog.shelf:
            return next(iter(self.catalog.shelf.values()))
        wanted = {Text.label_key(self.order.category), Text.label_key(self.order.item)} - {""}
        for page in self.catalog.pages.values():
            for label in page.get("category_path") or []:
                if Text.label_key(label) in wanted:
                    return str(label)
        return ""
    def shelf_filters(self, count: int = 4) -> List[Dict[str, Any]]:
        label = self.store_label()
        if not label or not self.session.has("filter") or self.order.budget is None:
            return []
        out = []
        for share in (0.7, 0.45, 0.25, 0.12, 0.85, 0.55, 0.35, 0.18, 0.08, 0.04):
            tag = f"filter|{Text.lowered(label)}|{share}"
            if tag not in self.sent and len(out) < count and len(self.sent) < self.search_room:
                self.sent.add(tag)
                out.append(StoreSession.call("filter", category=label, max_price=round(self.order.budget * share, 2), in_stock=True, k=10))
        return out
    def queue_filters(self) -> None:
        brand = next((r["value"] for r in self.order.requirements if "brand" in Text.lowered(r["property"])), None)
        label = self.store_label()
        for category in ([label] if label else self.label_guesses()[:2]):
            self.pending_filters.append(StoreSession.call("filter", category=category, brand=brand, max_price=self.order.budget, in_stock=True, k=10))
        if brand:
            self.pending_filters.append(StoreSession.call("filter", brand=brand, max_price=self.order.budget, in_stock=True, k=10))
    def next_searches(self, room: int) -> List[Dict[str, Any]]:
        out = self.pending_filters[:room] if self.session.has("filter") else []
        self.pending_filters = self.pending_filters[len(out):] if out else []
        while (self.planned or self.queries and len(self.sent) < self.search_room) and len(out) < room:
            query = (self.planned or self.queries).pop(0)
            if query in self.sent:
                continue
            self.sent.add(query)
            self.searched.append(query)
            out.append(StoreSession.call("search", query=" ".join(query.replace('"', " ").split()), k=10, in_stock=True))
        return out
    def stale(self) -> bool:
        log = self.catalog.fresh_log[-STALE_TURNS:]
        return len(log) >= STALE_TURNS and all(new < STALE_SHARE * max(1, shown) for new, shown in log)
    def plan_searches(self, why: str = "") -> List[str]:
        if (self.search_asks >= SEARCH_ROUNDS or not self.desk.can_think() or self.clock.late() or self.session.turns_left() <= TURN_RESERVE + 2
                or self.planned_at == self.session.turn):
            return []
        self.search_asks += 1
        self.planned_at = self.session.turn
        seen = list(dict.fromkeys(str(p.get("title") or "")[:90] for p in self.catalog.pages.values()))[-25:]
        fitting = any(k not in self.sieve.gone for k in self.sieve.ranked())
        text = (f"{self.order.state_text()}\n\nSearches tried:\n" + "\n".join(self.searched[-40:])
                + ("\n\nTitles seen:\n" if fitting else "\n\nTitles seen (none meets every requirement):\n") + "\n".join(seen)
                + (f"\n\n{why}" if why else ""))
        answers = self.desk.ask_all(SEAT_SEARCH, text)
        lists = [[q for q in self.desk.listed_fields(answer)["queries"] if isinstance(q, str) and q.strip()][:12] for answer in answers]
        taken = {Text.word_set(q) for q in list(self.sent) + self.planned}
        out = []
        for place in range(max((len(one) for one in lists), default=0)):
            for one in lists:
                query = Text.lowered(" ".join(one[place].split())[:QUERY_LENGTH]) if place < len(one) else ""
                words = Text.word_set(query)
                if words and words not in taken:
                    taken.add(words)
                    out.append(query)
        return out
    def send_planned(self) -> bool:
        spent = False
        for _ in range(PLAN_TURNS):
            if not self.planned or self.session.over or self.session.turns_left() <= TURN_RESERVE + 1 or self.clock.late():
                break
            half = self.session.group_size // 2
            opens = self.to_open(half * OPEN_BATCH)[:half]
            searches = self.next_searches(self.session.group_size - len(opens))
            if not searches:
                break
            self.session.send(opens + searches)
            self.reader.hear_quietly()
            self.judge.settle()
            spent = True
        if spent and not self.session.over and self.session.turns_left() > TURN_RESERVE and not self.clock.late():
            opens = self.to_open(self.session.group_size * OPEN_BATCH, allowance=len(self.catalog.pages) + 60)
            if opens:
                self.session.send(opens[: self.session.group_size])
                self.reader.hear_quietly()
            self.judge.settle()
        return spent
    def rethink_searches(self, why: str = "") -> bool:
        if (self.search_asks >= SEARCH_ROUNDS or self.session.turns_left() <= TURN_RESERVE + 2 or self.clock.late() or not self.desk.can_think()
                or self.planned_at == self.session.turn):
            return False
        planned = self.plan_searches(why)
        taken = {Text.word_set(q) for q in list(self.sent) + self.planned + planned}
        extra = []
        for query in self.shelf_queries(4) + self.kin_queries():
            query = Text.lowered(" ".join(str(query or "").split())[:QUERY_LENGTH])
            if query and Text.word_set(query) not in taken and len(extra) < 6:
                taken.add(Text.word_set(query))
                extra.append(query)
        if not planned and not extra:
            return False
        self.planned = planned + self.planned + extra
        return self.send_planned()
    def round_searches(self, room: int, fitting: List[Any], enough: int) -> List[Dict[str, Any]]:
        if self.fresh_rounds <= 0 and not self.planned and self.stale():
            self.planned = self.plan_searches("The latest searches brought back mostly listings already seen.")
        told_new = self.shelf_queries(8) if self.fresh_rounds > 0 and len(fitting) < enough else []
        if told_new:
            self.fresh_rounds -= 1
            named = [q for q in (Text.lowered(" ".join(str(v).split())[:QUERY_LENGTH]) for v in self.printed_queries() + self.near_kin_queries() + self.value_queries()) if q and q not in self.sent]
            named = list(dict.fromkeys(named))[: room // 2]
            self.add_queries(told_new, first=True)
            self.queries = named + [q for q in self.queries if q not in named]
            return self.next_searches(room)
        self.add_queries(self.kin_queries(), first=True)
        self.add_queries(self.printed_queries(), first=True)
        if len(fitting) < enough:
            self.add_queries(self.variant_queries(), first=True)
        if len(self.queries) < room // 2:
            self.add_queries(self.shelf_queries(6))
        searches = [] if self.planned else self.shelf_filters(room // 2)
        return searches + self.next_searches(room - len(searches))
    def row_score(self, key: Key) -> float:
        row = self.catalog.rows.get(key) or {}
        title = Text.stems(row.get("title"))
        score = 4.0 if key in self.catalog.shelf else 0.0
        for requirement in self.order.requirements:
            need = Text.stems(requirement["value"])
            others = [Text.stems(form) for form in list(requirement.get("written") or []) + list(requirement.get("alike") or [])]
            if need and need <= title or any(other and other <= title for other in others):
                score += 3
            elif need & title:
                score += 1
        score += sum(1 for w in {Text.singular(w) for w in self.item_words()} if w in title)
        if Text.lowered(row.get("title")) in self.kin_titles() or Text.lowered(row.get("title")) in self.probed:
            score += 20
        elif "-" in key[1] and key[1].rsplit("-", 1)[0] in {k[1].rsplit("-", 1)[0] for k in self.judge.verdicts if "-" in k[1] and self.sieve.fits_words(k)}:
            score += 20
        return score + (50 if self.order.listed and key in self.order.listed else 0)
    def to_open(self, limit: int, allowance: Optional[int] = None) -> List[Dict[str, Any]]:
        allowance = self.open_room if allowance is None else allowance
        unopened = [k for k in self.catalog.row_order if k[0] not in self.catalog.pages and (self.sieve.affordable(k) or self.sieve.just_over(k, OPEN_REACH))]
        if self.order.listed:
            unopened = [k for k in self.order.listed if k[0] not in self.catalog.pages] + [k for k in unopened if k in self.order.listed]
        fresh = (lambda k: self.catalog.shown_at.get(k, 0) >= self.order.changed_at) if self.order.changed_at else (lambda k: False)
        ranked = sorted(dict.fromkeys(unopened), key=lambda k: (not fresh(k), -(self.row_score(k) - (0 if self.sieve.affordable(k) else 2))))
        most = min(limit, OPEN_CALLS * OPEN_BATCH)
        ids = list(dict.fromkeys(k[0] for k in ranked))[: max(0, min(most, allowance - len(self.catalog.pages)))]
        if len(ids) < most and len(self.catalog.pages) < allowance + OPEN_SPARE:
            spare = [k[0] for k in ranked if k[0] not in ids and self.sieve.affordable(k) and self.carries_all(k)]
            ids += list(dict.fromkeys(spare))[: min(most - len(ids), allowance + OPEN_SPARE - len(self.catalog.pages))]
        if self.session.has("compare"):
            return [StoreSession.call("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)]
        return [StoreSession.call("view", product_id=i) for i in ids]
    def carries_all(self, key: Key) -> bool:
        if not self.order.requirements:
            return False
        title = Text.stems((self.catalog.rows.get(key) or {}).get("title"))
        for r in self.order.requirements:
            forms = [Text.stems(r["value"])] + [Text.stems(f) for f in list(r.get("written") or []) + list(r.get("alike") or [])]
            if not any(f and f <= title for f in forms):
                return False
        return True
    def rows_for(self, key: Key) -> List[Dict[str, Any]]:
        weak = [r for r in self.order.requirements if self.sieve.reliance(key, r) < SURE_FIT]
        if not weak:
            return []
        unopened = [k for k in self.catalog.row_order if k[0] not in self.catalog.pages and self.sieve.affordable(k)]
        found = []
        for k in unopened:
            title = Text.stems((self.catalog.rows.get(k) or {}).get("title"))
            for r in weak:
                forms = [Text.stems(r["value"])] + [Text.stems(f) for f in list(r.get("written") or []) + list(r.get("alike") or [])]
                if any(f and f <= title for f in forms):
                    found.append(k)
                    break
        ids = list(dict.fromkeys(k[0] for k in found))[: OPEN_CALLS * OPEN_BATCH]
        if not ids:
            return []
        if self.session.has("compare"):
            return [StoreSession.call("compare", product_ids=ids[i:i + OPEN_BATCH]) for i in range(0, len(ids), OPEN_BATCH)]
        return [StoreSession.call("view", product_id=i) for i in ids[:OPEN_CALLS]]
    def widen(self) -> None:
        self.widened = True
        for requirement in self.order.requirements:
            if not self.desk.can_think():
                break
            answer = self.desk.ask(SEAT_WORDS, f"Item: {self.order.item}\nProperty: {requirement['property']}\nValue: {requirement['value']}") or {}
            words = [str(w).strip() for w in answer.get("words") or [] if str(w).strip()][:6]
            figures = set(re.findall(r"\d+(?:\.\d+)?", requirement["value"]))
            spelt = [w for w in words if figures and figures <= set(re.findall(r"\d+(?:\.\d+)?", w))]
            for name, found in (("written", spelt), ("alike", [w for w in words if w not in spelt])):
                if found:
                    requirement[name] = list(dict.fromkeys(list(requirement.get(name) or []) + found))
            others = " ".join(Text.closed_up(r["value"]) for r in self.order.requirements if r is not requirement)
            head = (self.item_words() or [""])[-1]
            spelled = [Text.closed_up(word) for word in words[:3]]
            self.add_queries([f"{others} {word} {self.order.item}" for word in spelled] + [f"{others} {word} {head}" for word in spelled[:2]]
                             + [f"{others} {word}" for word in spelled[:2] if others], first=True)


ASK_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in two or three sentences: each reason for buying, what it bears on for this item, and why its first property leads."},
    "compared": {"type": "string", "description": "The property that the request's way of comparing fitting items is about ('the longest' is about the length), in plain words; empty when it gives none."},
    "reasons": {"type": "array", "maxItems": 2,
                "description": "One entry for each reason for buying the request gives, at most two, in the request's order; a single entry with an empty reason when it gives none.",
                "items": Schema.fields({
                    "reason": {"type": "string", "description": "The use or the person, in the request's own words; never how fitting items are compared, never a requirement it spells out."},
                    "bears_on": {"type": "string", "enum": ["appearance", "quantity", "dimension", "compatibility", "unstated"],
                                 "description": "What the reason asks of the item."},
                    "ask": Schema.strings("Four to six properties to ask about for this reason, the most likely first, each one to three plain words.")},
                    ["reason", "bears_on", "ask"])},
    "labels": Schema.strings("Each fact label shown above that names a property of the item itself, written '<the label exactly as shown> = <the property as a person reads it>': the label's words apart, every word kept, a word written short spelled out in full, a label like <x>_type or type_of_<x> read as '<x> type', no unit and no value. Leave out every label about who makes or sells the item (a brand, a model, a series), a guarantee, its condition, an id or a link, a description, or how the page is laid out."),
    "measures": Schema.strings("For each unit listed under the amounts that measures something of the item itself, '<the unit exactly as shown> = <what a figure in it measures>', in one to three plain words a shopper uses for that measure (how many come together is the number of pieces); leave out a unit that measures nothing of the item.")},
    ["reason", "compared", "reasons", "labels", "measures"])

ASK_INSTRUCTIONS = """You choose which properties a shopping assistant asks the shopper about. The shopper wants one item and, for every reason the request gives for buying it, keeps one requirement back: a value of a single property of the item. That value comes out only when a question names that very property. Questions about other properties, and broad questions with no property in them, bring nothing, and each question after the first lowers the shopper's opinion of the assistant. Your order is the order of the questions.

INPUT
The request and what is known of it, the questions already put with their outcome, and the store's view of this kind of item: some fitting listings, the fact labels its listings carry (each with how many listings carry it and its most common values), and the units their names and options use for amounts.

OUTPUT
reason first. Then compared: the property the request's way of comparing fitting items is about. Then reasons: one entry per reason for buying, each with what it bears on and four to six properties to ask about for it, most likely first. Then labels: the fact labels shown that name a property of the item itself, each with the property as a person reads it. Then measures: what each unit of the amounts measures.

WHICH WORDS ARE A REASON FOR BUYING
The use the item is bought for, or the person who will have it, in the request's own words. One clause gives one reason, however many things it mentions. How the shopper wants fitting items compared (the greatest, the longest, the most of something) is no reason for buying, and neither is any requirement the request spells out. Never add a reason the request does not give.

WHAT A REASON BEARS ON - judged from what the reason says the item will do or whom it will please, never from the kind of item
- appearance: the item's look, or pleasing the person who gets it. Ask the colour, then the material, then the pattern or the style.
- quantity: how much comes in the pack, or how long the supply lasts. Ask the number of pieces, then the weight, then the volume.
- dimension: the size it must have or the room it must fit into. Ask the measure of that size; usually the length.
- compatibility: equipment it is used with or plugs into, or an item the shopper already has that it must match or take the place of. Ask what has to agree with that equipment or item: the link it works through (the voltage, the plug or socket type), the kind of item as these listings' type labels name it, then its style, its colour, the size it must share.
- unstated: the reason names no use and no person to please, or says only that the choice was made already - by a wish someone stated, a list or a plan someone made, a habit of buying the same thing. Such a reason tells nothing of what the item must do or be, so the held-back value can be of any property: ask the colour first, then the properties these listings differ in most, by their own fact labels in plain words (the material among them), then the measures their names state in several figures.

ORDERING THE PROPERTIES OF ONE REASON
1. Begin with what the reason bears on, in the words these listings use for it. When the request already fixes the first of those, begin with the next one along the same line (for a quantity, the number of pieces when the weight is fixed); what the reason bears on stays the same.
2. Continue with properties the listings vary in: a fact label that shows several values, or a unit in which their names give several figures. A held-back value matters only where it separates the fitting listings, so a property they all share, or that few of them state, is seldom the one. Call a measure by what it measures (its length, its weight, how many pieces, its volume), never by its unit.
3. Skip what the request already fixes, the property in compared under any of its names (when the most pieces are wanted, the number in the pack too), the price, the stock, the category, the maker or seller, any guarantee or the item's condition, and anything already asked.
4. When there are two reasons, give them different first properties.
5. Always fill each list to four entries at least, from the properties these listings differ in: a short list leaves nothing to ask when the first guesses miss.

NAMING
A property is a few plain words (three at most), taken from a plain fact label (a label like <x>_type is the '<x> type'); each entry is a single property. A label written short or run together is said in full plain words, every word of it kept."""

SEAT_ASK = ModelSeat('ask_about', 'The properties of the item worth asking the shopper about.', ASK_INSTRUCTIONS, ASK_SCHEMA, 2048, (GLM, GLM, LUNA))


class Interviewer(Part):
    def __init__(self) -> None:
        self.questions_asked = 0
        self.awaiting_answer = False
        self.pair_open = False
        self.pair_again = False
        self.property_asks = 0
        self.to_ask: List[str] = []
        self.asked: List[str] = []
        self.asked_log: List[Key] = []
        self.current_ask = ""
        self.current_other = ""
        self.current_names: List[str] = []
        self.asked_wide: List[str] = []
        self.narrow = 0
        self.again: List[str] = []
        self.held_back = 0
        self.measure_units: Dict[str, Set[str]] = {}
        self.by_reason: List[List[str]] = []
        self.reason_words: List[Set[str]] = []
        self.needs_found = 0
        self.needs_expected = 1
        self.shelf_reads = 0
        self.shelf_line: Set[str] = set()
        self.shown: Set[str] = set()
        self.judged: Dict[str, str] = {}
        self.measures: Dict[str, str] = {}
        self.compared: Set[str] = set()
        self.colour_first = False
        self.reasked: Set[str] = set()
        self.last_question = ""
        self.plain_pending = False
        self.plain_asked = False
        self.use_unheard = False
        self.answers: List[str] = []
        self.spoke_of_use = False
    def queue_properties(self, names: List[str], compared: bool = False) -> None:
        fresh: List[str] = []
        settled = [Text.name_words(r["property"]) for r in self.order.requirements + self.order.set_aside]
        taken = [Text.name_words(p) for p in self.asked + self.to_ask]
        for name in names:
            raw = re.sub(r"^(?:the|an?|its)\s+", "", Text.lowered(str(name).replace("_", " ")))
            if re.search(r"[&/,+]|\b(?:and|or)\b", raw):
                continue
            text = " ".join(Text.word_list(raw))
            words = Text.name_words(text)
            if not words or len(text.split()) > 4 or NOT_A_PROPERTY.search(text) or "brand" in words:
                continue
            if any(len(Text.content_words(text) & Text.content_words(purpose)) >= 2 for purpose in self.order.purposes):
                continue
            if not compared and Text.content_words(text) & Text.content_words(self.order.priority):
                continue
            if any(words == other for other in taken + [Text.name_words(f) for f in fresh]):
                continue
            if any(words <= other or other <= words for other in settled if other):
                continue
            beside = next((i for i, f in enumerate(fresh) if len(words) >= 2 and len(Text.name_words(f)) >= 2
                           and (words < Text.name_words(f) or Text.name_words(f) < words)), None)
            if beside is None:
                fresh.append(text)
            else:
                fresh.insert(beside + 1, text)
        self.to_ask += fresh
    def leave_reason(self, name: str, line: str = "") -> None:
        if not name or len(self.by_reason) < 2:
            return
        words = lambda n: frozenset(Text.name_words(n))
        key = words(name)
        holding = [i for i, row in enumerate(self.by_reason) if any(words(n) == key for n in row)]
        said = Text.content_words(line)
        echo = {i: len(said & self.reason_words[i]) if i < len(self.reason_words) else 0 for i in range(len(self.by_reason))}
        best = max(echo.values())
        spoken = [i for i in echo if echo[i] == best] if best else []
        if len(spoken) == 1:
            index = spoken[0]
        elif holding:
            index = min(holding, key=lambda i: (next(j for j, n in enumerate(self.by_reason[i]) if words(n) == key), i))
        else:
            return
        self.by_reason.pop(index)
        if index < len(self.reason_words):
            self.reason_words.pop(index)
        wanted = [words(n) for row in self.by_reason for n in row]
        first = sorted((n for n in self.to_ask if words(n) in wanted), key=lambda n: wanted.index(words(n)))
        self.to_ask = first + [n for n in self.to_ask if words(n) not in wanted]
    def open_names(self) -> List[str]:
        known = self.asked + [r["property"] for r in self.order.requirements + self.order.set_aside]
        return [n for n in self.to_ask if not any(Text.name_words(n) == Text.name_words(other) for other in known)]
    def next_property(self) -> str:
        self.refresh_tail()
        open_names = self.open_names()
        name = (open_names or [""])[0]
        self.to_ask = [n for n in self.to_ask if n in open_names and n != name]
        if name:
            self.asked.append(name)
        return name
    def next_question(self) -> str:
        if (not self.needs_found and self.order.purposes and not self.plain_asked and not self.use_unheard
                and self.property_asks >= ASK_BLIND):
            self.current_names, self.current_ask, self.current_other = [], "", ""
            return PLAIN_QUESTION
        name = self.next_property()
        self.current_names = [name] if name else []
        self.property_asks += bool(name)
        self.current_ask, self.current_other = name, ""
        return ASK_ALONE.format(Text.spoken(name)) if name else ""
    def go_over(self) -> bool:
        again = [n for n in dict.fromkeys(self.asked_wide) if n in self.asked][: 2 * ASK_NARROW]
        if self.narrow or not again or self.needs_found:
            return False
        self.narrow = 1
        self.asked = [p for p in self.asked if p not in again]
        self.to_ask = again + [n for n in self.to_ask if n not in again]
        return True
    def put(self, question: str) -> Dict[str, Any]:
        self.last_question = question
        self.pair_open = bool(self.order.pair_question) and question == self.order.pair_question
        self.plain_asked = self.plain_asked or question == PLAIN_QUESTION
        self.spoke_of_use = question == PLAIN_QUESTION or USE_WORDS in question
        self.questions_asked += 1
        self.awaiting_answer = True
        return StoreSession.call("message", content=question[:400])
    def listen(self) -> Tuple[bool, bool]:
        if self.awaiting_answer and not self.catalog.heard:
            self.awaiting_answer = False
            self.answered("", 0, spoke_of_else=False)
        return self.reader.hear()
    def wants_to_ask(self) -> bool:
        if self.narrow > ASK_NARROW:
            return False
        if self.property_asks >= ASK_BLIND and (self.needs_found or self.plain_asked or self.use_unheard or not self.order.purposes):
            return False
        return (self.needs_found < self.needs_expected and self.questions_asked < ASK_LIMIT and self.session.has("message")
                and not self.session.over and self.session.turns_left() > TURN_RESERVE + 2 and not self.clock.late())
    def shelf_properties(self) -> str:
        pages = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        fitting = [k for k in self.judge.verdicts if self.sieve.fits_words(k) and self.sieve.affordable(k)]
        shown, titles = [], set()
        for key in fitting + [k for k in self.judge.verdicts if k not in fitting and (self.judge.verdicts[k] or {}).get("kind_ok")]:
            title = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title"))
            if (title, self.catalog.variant_of(key).get("options")) in titles or len(shown) >= 12:
                continue
            if len(shown) >= 4 and key not in fitting:
                break
            titles.add((title, self.catalog.variant_of(key).get("options")))
            shown.append(key)
        labels: Dict[str, Dict[str, int]] = {}
        for page in pages:
            for label, value in (page.get("facts") or {}).items():
                seen = labels.setdefault(str(label), {})
                for entry in Text.entries(value)[:4]:
                    seen[entry] = seen.get(entry, 0) + 1
        lines = []
        ranked = sorted(labels.items(), key=lambda kv: (-(len(kv[1]) > 1), -sum(kv[1].values()), kv[0]))
        self.shown = {label for label, _ in ranked[:ASK_SHELF]}
        for label, values in ranked[:ASK_SHELF]:
            top = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            lines.append(f"- {label} ({sum(values.values())} listings): " + ", ".join(v[:24] for v, _ in top))
        cards = "\n".join(self.judge.card(i, key) for i, key in enumerate(shown, start=1)) or "(none read yet)"
        units = "\n".join(f"- {unit} ({count} listings, {figures} different figures)" for unit, count, figures in self.shelf_units()) or "(none)"
        return (f"Listings that meet what is known so far:\n{cards}\n\n"
                f"Fact labels on the {len(pages)} listings of this kind read:\n" + ("\n".join(lines) or "(none)")
                + f"\n\nAmounts the names and options of these listings state, by unit:\n{units}")
    def shelf_units(self) -> List[Tuple[str, int, int]]:
        pages = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        known = {Text.unit_word(m.group(2)) for r in self.order.requirements + self.order.set_aside for form in self.matcher.forms(r)
                 for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", form)}
        seen: Dict[str, Tuple[Set[int], Set[float]]] = {}
        for index, page in enumerate(pages):
            text = " / ".join([Text.lowered(page.get("title"))] + [Text.lowered(v.get("options")) for v in page.get("variants") or []])
            for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s?([a-z]{1,10})\b", text):
                unit = Text.unit_word(m.group(2))
                listings, figures = seen.setdefault(unit, (set(), set()))
                listings.add(index)
                figures.add(float(m.group(1)))
        rows = [(unit, len(listings), len(figures)) for unit, (listings, figures) in seen.items()
                if unit not in known and len(listings) >= 3 and len(figures) >= 2]
        return sorted(rows, key=lambda row: (-row[1], row[0]))[:8]
    def shelf_labels(self) -> Dict[str, Tuple[int, Set[str], int]]:
        pages = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        seen: Dict[str, Tuple[int, Set[str], int]] = {}
        for page in pages:
            maker = Text.lowered(page.get("brand"))
            for label, value in (page.get("facts") or {}).items():
                count, entries, made = seen.get(str(label), (0, set(), 0))
                seen[str(label)] = (count + 1, entries | set(Text.entries(value)[:4]), made + bool(maker and Text.lowered(value) == maker))
        return seen
    def fitting_keys(self) -> List[Key]:
        judged = [k for k in self.judge.verdicts if self.sieve.fits_words(k) and self.sieve.affordable(k)]
        if len({k[0] for k in judged}) >= FIT_BASE:
            return judged
        shelf = [pid for pid, page in self.catalog.pages.items() if self.catalog.in_category(page)] or list(self.catalog.pages)
        variants = [(pid, str(v.get("sku"))) for pid in shelf for v in (self.catalog.pages[pid].get("variants") or []) if isinstance(v, dict)]
        stating = [k for k in variants if self.sieve.affordable(k) and all(self.matcher.settled(k, r) for r in self.order.requirements)]
        return stating if len({k[0] for k in stating}) >= FIT_BASE else variants
    def label_split(self, label: str, keys: List[Key]) -> int:
        carriers, entries = 0, set()
        for pid in {k[0] for k in keys}:
            value = ((self.catalog.pages.get(pid) or {}).get("facts") or {}).get(label)
            if value is not None and str(value).strip():
                carriers += 1
                entries |= set(Text.entries(value)[:4])
        return carriers if len(entries) >= 2 else 0
    def colour_split(self, keys: List[Key]) -> Tuple[int, int]:
        pages = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        colours: Set[str] = set()
        for page in pages:
            for label, value in (page.get("facts") or {}).items():
                if re.search(r"colou?r", str(label)):
                    colours |= {e for e in Text.entries(value) if re.fullmatch(r"[a-z]{3,}", e)}
            for variant in page.get("variants") or []:
                whole = Text.lowered(variant.get("options")) if isinstance(variant, dict) else ""
                if re.fullmatch(r"[a-z]{3,}", whole):
                    colours.add(Text.singular(whole))
        named: Dict[str, Set[str]] = {}
        for key in keys:
            page = self.catalog.pages.get(key[0]) or {}
            words = {Text.singular(w) for w in Text.word_list(self.catalog.variant_of(key).get("options"))}
            for label, value in (page.get("facts") or {}).items():
                if re.search(r"colou?r", str(label)):
                    words |= set(Text.entries(value))
            if words & colours:
                named.setdefault(key[0], set()).update(words & colours)
        return len(named), len(set().union(*named.values()) if named else set())
    def unit_split(self, keys: List[Key]) -> Dict[str, int]:
        seen: Dict[str, Tuple[Set[str], Set[float]]] = {}
        for key in keys:
            page = self.catalog.pages.get(key[0]) or {}
            text = Text.lowered(page.get("title")) + " / " + Text.lowered(self.catalog.variant_of(key).get("options"))
            for m in re.finditer(r"(?<![\w.])(\d+(?:\.\d+)?)\s?([a-z]{1,10})\b", text):
                listings, figures = seen.setdefault(Text.unit_word(m.group(2)), (set(), set()))
                listings.add(key[0])
                figures.add(float(m.group(1)))
        return {unit: len(listings) for unit, (listings, figures) in seen.items() if len(figures) >= 2}
    def label_tail(self, ahead: List[str]) -> List[str]:
        keys = self.fitting_keys()
        spread = self.shelf_labels()
        scored: List[Tuple[int, int, str]] = []
        for label, (count, entries, made) in spread.items():
            if len(entries) < 2 or 2 * made > count or self.counts_priority(entries) or re.search(r"colou?r", label):
                continue
            if label in self.shown and label not in self.judged:
                continue
            if sum(len(e) for e in entries) > 40 * len(entries):
                continue
            name = self.judged.get(label) or Text.label_name(label, entries)
            if not OF_THE_SALE.search(name):
                scored.append((self.label_split(label, keys), count, name))
        splits = self.unit_split(keys)
        for unit, count, _ in self.shelf_units():
            if self.measures.get(unit):
                scored.append((splits.get(unit, 0), count, self.measures[unit]))
        if not self.colour_first:
            listings, colours = self.colour_split(keys)
            scored.append((listings if colours >= 2 else 0, 0, "colour"))
        ordered = [name for _, _, name in sorted(scored, key=lambda row: (-row[0], -row[1]))]
        ahead_words = [Text.name_words(n) for n in ahead]
        tail: List[str] = []
        for name in ordered:
            words = Text.name_words(name)
            if not words or (self.compared and words == self.compared) or any(words <= w or w <= words for w in ahead_words):
                continue
            if not any(words < Text.name_words(other) for other in ordered):
                tail.append(name)
        return tail
    def refresh_tail(self) -> None:
        if not self.shelf_reads:
            return
        head = [n for n in self.to_ask if n not in self.shelf_line]
        self.to_ask = head
        self.queue_properties(self.label_tail(head))
        self.shelf_line = set(self.to_ask[len(head):])
    def read_labels(self, labels: Any) -> None:
        self.judged = {}
        entries = [str(entry) for entry in labels] if isinstance(labels, list) else []
        spread = self.shelf_labels()
        found: List[Tuple[str, str]] = []
        for entry in entries:
            shown, _, plain = entry.partition("=")
            label = next((k for k in spread if Text.label_key(k.replace("_", " ")) == Text.label_key(shown.replace("_", " "))), None)
            if label is not None:
                found.append((label, plain))
        if not found or 2 * len(found) < len(entries):
            self.shown = set()
            return
        for label, plain in found:
            own_name = Text.label_name(label, spread[label][1])
            own, said = Text.name_words(own_name), " ".join(Text.word_list(plain))
            read = Text.name_words(said)
            kept = all(w in read or ((len(w) <= 3 or not re.search(r"[aeiou]", w)) and any(Text.shortened(w, r) for r in read)) for w in own)
            self.judged[label] = " ".join(Text.spoken(w) for w in said.split()) if said and kept and len(read) <= len(own) + 1 else own_name
    def read_measures(self, measures: Any) -> None:
        for entry in measures if isinstance(measures, list) else []:
            unit, _, name = str(entry).partition("=")
            words = Text.word_list(name)
            if unit.strip() and 1 <= len(words) <= 3 and words[0] not in ("none", "nothing") and not NOT_A_PROPERTY.search(" ".join(words)):
                self.measures[Text.unit_word(Text.lowered(unit).strip())] = " ".join(words)
    def counts_priority(self, entries: Set[str]) -> bool:
        said = Text.content_words(self.order.priority)
        counted = [e for e in entries if re.match(r"\d", e) and Text.content_words(e) & said]
        return bool(said) and 2 * len(counted) > len(entries)
    def counted_label(self, name: str) -> bool:
        spread = self.shelf_labels()
        return any(Text.name_words(k) == Text.name_words(name) and self.counts_priority(entries) for k, (_, entries, _) in spread.items())
    def in_full(self, name: str) -> str:
        words = Text.name_words(name)
        spread = self.shelf_labels()
        longer = [k for k in spread if words < Text.name_words(k) and len(Text.word_list(k.replace("_", " "))) <= 4
                  and all(len(w) >= 3 for w in Text.word_list(k.replace("_", " ")))]
        best = max(longer, key=lambda k: (spread[k][0], k), default="")
        return (self.judged.get(best) or Text.label_name(best, spread[best][1])) if best else name
    def read_shelf(self) -> None:
        if self.shelf_reads >= 3 or not self.catalog.pages or not self.desk.can_think() or (
                self.shelf_reads and len([n for n in self.open_names() if n not in self.shelf_line]) >= 3):
            return
        self.shelf_reads += 1
        self.judge.settle()
        asked = "\n".join(f"- {question} -> {result}" for question, result in self.asked_log) or "(nothing yet)"
        text = (f"{self.order.state_text()}\nReasons for buying named in the request: {'; '.join(self.order.purposes) or '(none)'}\n\n"
                f"Already asked:\n{asked}\n\n{self.shelf_properties()}")
        answer = self.desk.ask(SEAT_ASK, text) or {}
        entries = answer.get("reasons")
        entries = [e for e in (entries if isinstance(entries, list) else []) if isinstance(e, dict)][:2]
        if not entries:
            return
        known = [r["property"] for r in self.order.requirements + self.order.set_aside] + self.asked
        colour_known = any(re.search(r"colou?r", str(p), re.I) for p in known)
        compared = self.compared = Text.name_words(Text.lowered(str(answer.get("compared") or "")))
        self.read_labels(answer.get("labels"))
        self.read_measures(answer.get("measures"))
        rows: List[List[str]] = []
        for entry in entries:
            names = entry.get("ask")
            names = re.split(r"[\n;,]+", names) if isinstance(names, str) else (names if isinstance(names, list) else [])
            names = [self.in_full(Text.lowered(n)) for n in names if isinstance(n, str) and n.strip() and not re.search(r"colou?r", n, re.I)
                     and not (compared and Text.name_words(Text.lowered(n)) == compared) and not self.counted_label(Text.lowered(n))]
            names = [n for i, n in enumerate(names) if not any(Text.name_words(n) == Text.name_words(m) for m in names[:i])]
            rows.append(names[:ASK_ROW])
        self.by_reason = [list(row) for row in rows]
        self.reason_words = [Text.content_words(str(e.get("reason") or "")) for e in entries]
        listings, colours = self.colour_split(self.fitting_keys())
        uses = all(e.get("bears_on") in ("quantity", "dimension") for e in entries)
        self.colour_first = not colour_known and not uses and not (listings >= COLOUR_PROOF and colours == 1)
        in_turn = (["colour"] if self.colour_first else []) + [row[i] for i in range(ASK_ROW) for row in rows if i < len(row)]
        left = [n for n in self.open_names() if n not in self.shelf_line and not re.search(r"colou?r", n)]
        self.to_ask, self.shelf_line = [], set()
        self.queue_properties(in_turn + left, compared=True)
        self.refresh_tail()
    def find_out(self) -> None:
        opening = True
        while self.wants_to_ask():
            if self.episode.cart_early() or self.episode.reply_early():
                opening = True
                continue
            if not self.catalog.cart and self.episode.first_cart_due() and self.episode.cart_due():
                opening = True
                continue
            if self.catalog.pages:
                self.read_shelf()
            if not self.catalog.pages and self.catalog.rows and self.held_back < 2:
                self.held_back += 1
                question = ""
            elif self.plain_pending or (not self.needs_found and not self.plain_asked and not self.use_unheard
                                        and not self.order.purposes and self.questions_asked >= ASK_LIMIT - 1):
                question, self.plain_pending = PLAIN_QUESTION, False
            else:
                question = self.next_question()
                if not question and self.catalog.pages and self.shelf_reads < 4:
                    self.read_shelf()
                    question = self.next_question()
                if not question and self.catalog.pages and self.go_over():
                    question = self.next_question()
            if not question and self.catalog.pages and not self.needs_found and not self.plain_asked and not self.use_unheard:
                question = PLAIN_QUESTION
            if not question:
                if self.catalog.pages or not self.catalog.rows:
                    break
                question = ""
            work: List[Dict[str, Any]] = []
            room = self.session.group_size - (1 if question else 0)
            if opening:
                work = self.planner.to_open(room * OPEN_BATCH)[:room]
            if not work:
                work = self.planner.shelf_filters(room // 2)
                work += self.planner.next_searches(room - len(work))
                if not work:
                    work = self.planner.to_open(room * OPEN_BATCH)[:room]
            opening = not opening
            if not question and not work:
                break
            calls = ([self.put(question)] if question else []) + work
            self.session.send(calls)
            self.listen()
    def answered(self, line: str, given: int, spoke_of_else: bool, about: str = "", nothing: bool = False) -> None:
        question, name, other = self.last_question, self.current_ask, self.current_other
        named = list(self.current_names) or [n for n in (name, other) if n]
        self.current_ask = self.current_other = ""
        self.current_names = []
        if line and line in self.answers:
            self.asked_log.append((question, "the same answer as before"))
            self.use_unheard = True
            return
        if given and not spoke_of_else:
            self.answers.append(line)
            said = Text.name_words(line)
            told = self.order.requirements[-given:]
            kinds = {self.matcher.kind(r) for r in told}
            units = {Text.unit_word(m.group(2)) for r in told for form in self.matcher.forms(r) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", form)}
            def about_this(n: str) -> bool:
                words = Text.name_words(n)
                if words <= said or any(words <= Text.name_words(r["property"]) or Text.name_words(r["property"]) <= words for r in self.order.requirements):
                    return True
                if ("colour" in kinds and re.search(r"colou?r", n)) or ("material" in kinds and "material" in n):
                    return True
                return bool(self.measure_units.get(n, set()) & units) or " ".join(Text.word_list(n)) == about
            group = named if len(named) >= 2 else []
            unsaid = [n for n in group if not about_this(n)]
            by_the_use = bool(group) and len(unsaid) == len(group) and self.spoke_of_use
            apart = bool(group) and len(unsaid) == len(group) and not by_the_use
            for n in unsaid:
                self.asked = [p for p in self.asked if p != n]
                self.again.append(n)
            if self.spoke_of_use:
                self.use_unheard = True
            choice = self.pair_open and any(r.get("chosen") for r in told)
            if self.pair_open and not choice and not self.pair_again and not self.order.pair_chosen:
                self.pair_again = True
            self.asked_log.append((question, f"the shopper said: {line[:160]}"))
            if choice:
                self.order.pair_chosen = True
            else:
                self.needs_found += 1
                if question == PLAIN_QUESTION and self.order.told_to_ask:
                    self.needs_expected = self.needs_found
                self.leave_reason(name, line)
            rest = [n for n in self.to_ask if n not in self.again]
            if by_the_use:
                self.to_ask = self.again + rest
            elif apart:
                half = (len(self.again) + 1) // 2
                self.to_ask = rest[:1] + self.again[:half] + rest[1:2] + self.again[half:] + rest[2:]
            else:
                self.to_ask = rest[:2] + self.again + rest[2:]
            self.again = []
            return
        if spoke_of_else and question and (name or question) not in self.reasked:
            self.reasked.add(name or question)
            if name:
                for again in reversed(named):
                    self.asked = [p for p in self.asked if p != again]
                    self.to_ask.insert(0, again)
            elif question == PLAIN_QUESTION:
                self.plain_pending, self.plain_asked = True, False
            return
        if line and nothing and len(named) == 2 and question not in self.reasked and not self.reader.restates_budget(line):
            self.reasked.add(question)
            self.asked_log.append((question, "not answered"))
            self.asked = [p for p in self.asked if p not in named]
            self.to_ask = named + [n for n in self.to_ask if n not in named]
            self.use_unheard = True
            return
        self.asked_log.append((question, "no requirement"))
        if len(named) >= 3:
            self.asked_wide += named
        elif self.narrow and named:
            self.narrow += 1
        if line and self.spoke_of_use:
            self.use_unheard = True


class Matcher(Part):
    def __init__(self) -> None:
        self.own_cache: Dict[Tuple[Any, ...], Any] = {}
        self.own_cache_at: Tuple[Any, ...] = ()
        self.read_cache: Dict[str, Any] = {}
        self.read_stamp: Tuple[Any, ...] = ()
        self.part_cache: Dict[str, Any] = {}
        self.named_as: Dict[Tuple[Any, ...], str] = {}
        self.name_checks = 0
        self.lenient = False
        self.filed_cache: Dict[str, Tuple[Tuple[int, bool], Set[str]]] = {}
    def remembered(self, name: str, key: Key, requirement: Dict[str, Any], work: Any) -> Any:
        stamp = (self.catalog.page_writes, self.catalog.named_shelf(), self.order.category, self.order.item, self.lenient)
        if stamp != self.read_stamp:
            self.read_cache, self.read_stamp = {}, stamp
        tag = (name, key, repr(sorted(requirement.items())))
        if tag not in self.read_cache:
            self.read_cache[tag] = work()
        return self.read_cache[tag]
    @staticmethod
    def forms(requirement: Dict[str, Any], loose: bool = False) -> Set[str]:
        return set(Matcher.kept_forms(requirement, loose))
    @staticmethod
    def kept_forms(requirement: Dict[str, Any], loose: bool = False) -> FrozenSet[Any]:
        tag = ("forms", str(requirement["value"]), tuple(str(w) for w in requirement.get("written") or []),
               tuple(str(w) for w in requirement.get("alike") or []), tuple(str(w) for w in requirement.get("store") or []),
               bool(requirement.get("filed")), loose)
        done = Text.memo.get(tag)
        if done is None:
            done = Text.memo[tag] = frozenset(Matcher._forms(requirement, loose))
        return done
    @staticmethod
    def _forms(requirement: Dict[str, Any], loose: bool) -> Set[str]:
        spellings = list(requirement.get("written") or [])
        store = [str(w) for w in requirement.get("store") or []]
        if requirement.get("filed"):
            written = spellings + list(requirement.get("alike") or []) if loose else [requirement["value"]] + store
        elif re.search(r"\d", requirement["value"]):
            if re.search(r"\d\s*[a-z]", Text.lowered(requirement["value"])):
                spellings = [form for form in spellings if not re.fullmatch(r"[\d.,\s]+", str(form).strip()) and re.search(r"\d", str(form))
                             and Text.same_amount(requirement["value"], form)]
            written = list(requirement.get("alike") or []) if loose else [requirement["value"]] + spellings + store
        else:
            close = [form for form in spellings if SequenceMatcher(None, Text.lowered(form), Text.lowered(requirement["value"])).ratio() >= 0.75
                     and len(Text.closed_up(str(form))) >= len(Text.closed_up(requirement["value"])) - 1]
            written = [form for form in spellings if form not in close] + list(requirement.get("alike") or []) if loose else [requirement["value"]] + close + store
        return {" ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(form))) for form in written} - {""}
    def forms_under(self, requirement: Dict[str, Any], label: str) -> FrozenSet[Any]:
        base = self.kept_forms(requirement)
        whole = bool(base & self.filed_under(label))
        tag = ("under", base, label, whole)
        done = Text.memo.get(tag)
        if done is None:
            done = Text.memo[tag] = frozenset(self._forms_under(base, label, whole))
        return done
    def _forms_under(self, base: FrozenSet[Any], label: str, whole: bool) -> Set[str]:
        named = Text.name_words(label) | {"at", "the", "of", "a", "an", "in", "on", "for", "with", "its"}
        out = set(base)
        if whole:
            return out
        for form in list(out):
            words = form.split()
            kept = [w for w in words if not any(Text.same_word(w, n) for n in named)]
            if kept and len(kept) < len(words):
                out.add(" ".join(kept))
        return out
    def filed_under(self, label: str) -> Set[str]:
        pages = self.catalog.pages
        shelved = self.catalog.any_shelved()
        cached = self.filed_cache.get(label)
        if cached and cached[0] == (len(pages), shelved):
            return cached[1]
        found = {entry for page in pages.values() if not shelved or self.catalog.in_category(page)
                 for entry in Text.entries((page.get("facts") or {}).get(label))}
        self.filed_cache[label] = ((len(pages), shelved), found)
        return found
    def states(self, text: Any, requirement: Dict[str, Any]) -> bool:
        entries = {" ".join(Text.singular(w) for w in Text.word_list(reading(part))) for part in re.split(r"[,;/|]+|\s+-\s+", Text.lowered(text)) for reading in (Text.lowered, Text.closed_up)}
        return bool(entries & self.forms(requirement))
    def picks_kind(self, requirement: Dict[str, Any]) -> Optional[Set[str]]:
        if re.search(r"\d", requirement["value"]):
            return None
        words = Text.content_words(requirement["value"])
        kinds = [Text.content_words(part) for name in (self.order.item, self.catalog.named_shelf())
                 for part in re.split(r"\s*(?:&|,|/|\band\b)\s*", Text.lowered(name)) if Text.content_words(part)]
        if len(kinds) < 2 or not words or all(kind <= words for kind in kinds):
            return None
        return next((kind for kind in kinds if words <= kind), None)
    def kind(self, requirement: Dict[str, Any]) -> str:
        if re.search(r"\d", requirement["value"]):
            return "amount"
        name = Text.lowered(requirement["property"])
        if re.search(r"colou?r", name):
            return "colour"
        if "material" in name:
            return "fact" if self.part_labels(requirement) else "material"
        return "fact"
    def part_labels(self, requirement: Dict[str, Any]) -> Set[str]:
        stamp = (requirement["property"], len(self.catalog.pages))
        cached = self.part_cache.get(requirement["property"])
        if cached and cached[0] == stamp:
            return cached[1]
        wanted = Text.name_words(requirement["property"])
        part = wanted - Text.name_words("material type")
        found: Set[str] = set()
        if part:
            for page in self.catalog.pages.values():
                for label in page.get("facts") or {}:
                    have = Text.name_words(label)
                    if have and have <= wanted and have & part:
                        found.add(str(label))
        self.part_cache[requirement["property"]] = (stamp, found)
        return found
    def entry_says(self, entry: str, requirement: Dict[str, Any], label: str, loose: bool = False) -> bool:
        forms = self.kept_forms(requirement, loose=True) if loose else self.forms_under(requirement, label)
        tag = ("entry says", entry, forms)
        if tag not in Text.memo:
            have = entry.split()
            Text.memo[tag] = any(all(any(Text.same_word(n, h) for h in have) for n in form.split()) for form in forms)
        return Text.memo[tag]
    def entry_is(self, entry: str, requirement: Dict[str, Any], label: str) -> bool:
        kept = Text.memo.get(("kept", entry, label))
        if kept is None:
            named = Text.name_words(label)
            kept = Text.memo[("kept", entry, label)] = " ".join(w for w in entry.split() if not any(Text.same_word(w, n) for n in named)) or entry
        forms = self.forms_under(requirement, label)
        if entry in forms or kept in forms:
            return True
        return any(len(said.split()) == len(form.split()) and all(Text.same_word(a, b) for a, b in zip(said.split(), form.split()))
                   for said in {entry, kept} for form in forms if not re.search(r"\d", form))
    def own_labels(self, requirement: Dict[str, Any]) -> Tuple[Set[str], Set[str], Set[str]]:
        pages = self.catalog.pages
        if self.own_cache_at != (len(pages), self.catalog.named_shelf()):
            self.own_cache, self.own_cache_at = {}, (len(pages), self.catalog.named_shelf())
        stamp = (requirement["property"], requirement["value"], tuple(str(w) for w in requirement.get("written") or []),
                 tuple(str(w) for w in requirement.get("store") or []), tuple(str(w) for w in requirement.get("alike") or []),
                 bool(requirement.get("filed")))
        cached = self.own_cache.get(stamp)
        if cached is not None:
            return cached
        wanted = Text.name_words(requirement["property"])
        kind = self.kind(requirement)
        shelf = [page for page in pages.values() if self.catalog.in_category(page)] or list(pages.values())
        written = " ".join(Text.word_list(requirement["property"]))
        tiers: List[Set[str]] = [set(), set(), set(), set()]
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", form)}
        stated: Dict[str, int] = {}
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                have = Text.name_words(label)
                if wanted and have:
                    if " ".join(Text.word_list(str(label).replace("_", " "))) == written:
                        tiers[0].add(str(label))
                    elif have == wanted:
                        tiers[1].add(str(label))
                    elif have <= wanted or wanted <= have:
                        tiers[2 if have.symmetric_difference(wanted) <= units else 3].add(str(label))
                entries = Text.entries(value)
                if len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                    stated[str(label)] = stated.get(str(label), 0) + 1
        related = next((tier for tier in tiers if tier), set())
        if not related and len(wanted - {"type", "kind"}) >= 1:
            figures = bool(re.search(r"\d", requirement["value"]))
            for page in shelf:
                for label, value in (page.get("facts") or {}).items():
                    have = Text.name_words(label) - {"type", "kind"}
                    if (have & (wanted - {"type", "kind"})) and any(bool(re.search(r"\d", e)) == figures and (not figures or re.search(r"\d\s*[a-z]", e))
                                                                   for e in Text.entries(value)):
                        related.add(str(label))
        if kind != "amount":
            related = set().union(*tiers)
            if "material" in Text.lowered(requirement["property"]) and self.part_labels(requirement):
                related = self.part_labels(requirement)
        most = max(stated.values(), default=0)
        inferred: Set[str] = set()
        if kind != "colour" and (kind == "amount" or not related or not wanted - {"type", "kind"}) and not any(stated.get(label) for label in related):
            inferred = {label for label, count in stated.items() if count >= 2 and 3 * count >= most}
        words: Set[str] = set()
        for page in shelf:
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    words |= {entry for entry in Text.entries(value) if " " not in entry}
            if kind == "colour":
                for variant in page.get("variants") or []:
                    words |= {entry for entry in Text.entries(variant.get("options")) if re.fullmatch(r"[a-z]{3,}", entry)}
        answer = (related, inferred, words)
        self.own_cache[stamp] = answer
        return answer
    def named_facts(self, page: Dict[str, Any], requirement: Dict[str, Any]) -> List[Key]:
        related, inferred, _ = self.own_labels(requirement)
        return [(str(label), str(value)) for label, value in (page.get("facts") or {}).items() if str(label) in related or str(label) in inferred]
    @staticmethod
    def refused(requirement: Dict[str, Any], text: Any) -> bool:
        refused = requirement.get("refused")
        return bool(refused) and " ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(text))) in refused
    def says(self, text: Any, requirement: Dict[str, Any], loose: bool = False) -> bool:
        have = {Text.singular(w) for reading in (Text.lowered(text), Text.closed_up(text)) for w in Text.word_list(reading)}
        return any(all(any(Text.same_word(n, h) for h in have) for n in form.split()) for form in self.forms(requirement, loose))
    def amounts(self, text: Any, requirement: Dict[str, Any]) -> Tuple[Set[Tuple[float, str]], Set[Tuple[float, str]]]:
        tag = ("amounts", text if isinstance(text, str) else str(text or ""), str(requirement["value"]),
               tuple(str(w) for w in requirement.get("written") or []), tuple(str(w) for w in requirement.get("alike") or []),
               tuple(str(w) for w in requirement.get("store") or []), bool(requirement.get("filed")), self.lenient)
        done = Text.memo.get(tag)
        if done is None:
            same, other = self._amounts(text, requirement)
            done = Text.memo[tag] = (frozenset(same), frozenset(other))
        return set(done[0]), set(done[1])
    def _amounts(self, text: Any, requirement: Dict[str, Any]) -> Tuple[Set[Tuple[float, str]], Set[Tuple[float, str]]]:
        unit = Text.unit_word
        wanted: Set[Tuple[float, str]] = set()
        forms = self.forms(requirement)
        if self.lenient:
            forms = forms | {" ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(form))) for form in requirement.get("written") or []
                             if self.converted(requirement["value"], form)}
        for form in forms:
            wanted |= {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", form)}
        found = {(float(m.group(1)), unit(m.group(2))) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([a-z]+|\")(?![a-z0-9])", Text.lowered(text))}
        units = {name for _, name in wanted}
        long_forms = {name for name in units if len(name) >= 4}
        def fits(seen: str, name: str) -> bool:
            if len(seen) >= 4 and long_forms and seen not in long_forms:
                return len(name) >= 4 and (Text.same_unit(seen, name) or SequenceMatcher(None, seen, name).ratio() >= 0.75)
            return Text.same_unit(seen, name)
        found = {pair for pair in found if any(fits(pair[1], name) for name in units)}
        same = {pair for pair in found if any(pair[0] == number and fits(pair[1], name) for number, name in wanted)}
        return same, found - same
    @staticmethod
    def converted(value: str, form: Any) -> bool:
        wanted = [(float(m.group(1)), Text.unit_word(m.group(2))) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", Text.lowered(value))]
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", Text.lowered(form)):
            number, unit = float(m.group(1)), Text.unit_word(m.group(2))
            if any(abs(number - n) < 1e-9 and not Text.same_unit(unit, u) for n, u in wanted):
                return False
        return True
    def unit_in_label(self, label: str, requirement: Dict[str, Any]) -> bool:
        units = {m.group(2) for form in self.forms(requirement) for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", form)}
        return bool(units & Text.name_words(label))
    def named_otherwise(self, options: Any, requirement: Dict[str, Any]) -> bool:
        named = [w for w in Text.name_words(requirement["property"]) if len(w) >= 5]
        for part in re.split(r"[/,;|]+", Text.lowered(options)):
            if not re.search(r"\d", part) or self.has_amount(part, requirement):
                continue
            if any(SequenceMatcher(None, word, name).ratio() >= 0.8 for word in re.findall(r"[a-z]{4,}", part) for name in named):
                return True
        return False
    def has_amount(self, text: Any, requirement: Dict[str, Any]) -> bool:
        same, others = self.amounts(text, requirement)
        if same or (not others and self.says(text, requirement)):
            return True
        bare = re.fullmatch(r"\d+(?:\.\d+)?", requirement["value"].strip())
        if not bare:
            return False
        named = [w for w in Text.word_list(requirement["property"]) if len(w) >= 3]
        return any(float(m.group(1)) == float(bare.group(0)) and any(Text.same_unit(m.group(2), w) for w in named)
                   for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+|\")", Text.lowered(text)))
    @staticmethod
    def written_otherwise(entry: str, requirement: Dict[str, Any]) -> bool:
        said = " ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(entry)))
        return any(said == " ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(form))) and not Text.same_amount(requirement["value"], form)
                   for form in requirement.get("written") or [] if re.search(r"\d", str(form)))
    def last_side(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        side = r"\d+(?:\.\d+)?"
        for value in (page.get("facts") or {}).values():
            sizes = [part.strip() for part in re.split(r"[,;/|]+", Text.lowered(value)) if re.search(side + r"\s*[a-z]*\s*[x×*]\s*" + side, part)]
            if len(sizes) != 1:
                continue
            found = re.fullmatch(r"((?:" + side + r"\s*(?:(?!x)[a-z]+)?\s*[x×*]\s*)+)(" + side + r")\s*([a-z]+)", sizes[0])
            if found and self.amounts(found.group(2) + found.group(3), requirement)[0] and not self.amounts(value, requirement)[1]:
                return True
        return False
    def converted_on(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.remembered("converted_on", key, requirement, lambda: self._converted_on(key, requirement))
    def _converted_on(self, key: Key, requirement: Dict[str, Any]) -> bool:
        others = []
        for form in requirement.get("written") or []:
            found = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([a-z]+|\")\s*", Text.lowered(form))
            if found and not Text.same_amount(requirement["value"], form):
                others.append((float(found.group(1)), Text.unit_word(found.group(2))))
        if not others:
            return False
        page = self.catalog.pages.get(key[0]) or {}
        options = str(self.catalog.variant_of(key).get("options") or "")
        readings = []
        for text in (options, page.get("title")):
            if self.amounts(text, requirement)[0] or self.amounts(text, requirement)[1]:
                return False
            seen = {(float(n), Text.unit_word(m.group(2))) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*)\s*([a-z]+|\")", Text.lowered(text))
                    for n in re.findall(r"\d+(?:\.\d+)?", m.group(1))}
            mine = {(n, u) for n, u in seen if any(Text.same_unit(u, unit) for _, unit in others)}
            hits = {(n, u) for n, u in mine if any(Text.same_unit(u, unit) and abs(n - number) <= 0.025 * number for number, unit in others)}
            readings.append((mine, hits))
        (opt_mine, opt_hits), (title_mine, title_hits) = readings
        if opt_mine:
            return bool(opt_hits) and opt_mine == opt_hits and title_mine <= title_hits | opt_hits
        return bool(title_hits) and title_mine == title_hits
    def other_amount(self, text: Any, requirement: Dict[str, Any]) -> bool:
        same, others = self.amounts(text, requirement)
        return bool(others) and not same
    def plain(self, key: Key, requirement: Dict[str, Any]) -> int:
        return self.remembered("plain", key, requirement, lambda: self._plain(key, requirement))
    def _plain(self, key: Key, requirement: Dict[str, Any]) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        options = str(self.catalog.variant_of(key).get("options") or "")
        title = page.get("title")
        kind = self.kind(requirement)
        if kind == "amount":
            for token in {w for w in Text.word_list(page.get("brand")) if re.fullmatch(r"\d+[a-z]{1,2}", w)}:
                options = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", options)
                title = re.sub(rf"(?i)(?<![a-z0-9.]){token}(?![a-z0-9])", " ", str(title or ""))
        related, _, words = self.own_labels(requirement)
        own = [(label, Text.entries(value)) for label, value in self.named_facts(page, requirement)]
        singles = [(label, entries[0]) for label, entries in own if len(entries) == 1]
        lists = [(label, entry) for label, entries in own if len(entries) > 1 for entry in entries]
        stated = any(self.entry_says(entry, requirement, label) and not self.refused(requirement, entry) for label, entry in singles)
        listed = any(self.entry_says(entry, requirement, label) and not self.refused(requirement, entry) for label, entry in lists)
        value_words = {w for form in self.forms(requirement) for w in form.split()}
        elsewhere = any(self.says(value, requirement) for value in (page.get("facts") or {}).values())
        if kind == "amount":
            if self.other_amount(options, requirement) or self.named_otherwise(options, requirement):
                return -1
            number = re.search(r"\d+(?:\.\d+)?", requirement["value"])
            if self.has_amount(options, requirement):
                side = r"\d+(?:\.\d+)?\s*(?:(?!x)[a-z]+)?"
                apart = re.sub(side + r"(?:\s*[x×*]\s*" + side + r")+", " ", Text.lowered(options))
                return 1 if self.amounts(apart, requirement)[1] and self.has_amount(apart, requirement) else 3
            bare = [label for label, entry in singles if number and re.fullmatch(r"\d+(?:\.\d+)?", entry) and float(entry) == float(number.group(0))]
            if any(self.unit_in_label(label, requirement) for label in bare) or any(self.has_amount(entry, requirement) for _, entry in singles):
                if self.amounts(title, requirement)[1] and not self.amounts(options, requirement)[0] and not self.amounts(options, requirement)[1]:
                    return 1
                return 3
            if any(self.written_otherwise(entry, requirement) for _, entry in singles):
                return 2
            if bare:
                return 2 if self.has_amount(title, requirement) else 1
            if any(self.amounts(entry, requirement)[1] or (re.search(r"\d", entry) and self.unit_in_label(label, requirement)) for label, entry in singles):
                return -1
            if self.has_amount(title, requirement):
                return 1 if self.amounts(title, requirement)[1] else 2
            if self.other_amount(title, requirement) and not self.amounts(options, requirement)[0]:
                return -1
            for value in (page.get("facts") or {}).values():
                if self.amounts(value, requirement)[1]:
                    continue
                for entry in Text.entries(value):
                    if self.amounts(entry, requirement)[0] and len(re.findall(r"\d+(?:\.\d+)?", entry)) == 1:
                        return 2
            if self.converted_on(key, requirement) or self.last_side(key, requirement):
                return 1
            return 1 if listed or any(self.has_amount(value, requirement) for value in (page.get("facts") or {}).values()) else 0
        named_as = self.named_as.get((key, requirement["property"], requirement["value"]))
        if kind == "colour":
            wanted = self.forms(requirement)
            parts = [{Text.singular(w.strip(".")) for reading in (Text.lowered(part), Text.closed_up(part)) for w in re.split(r"[^a-z0-9.]+", reading) if w.strip(".")}
                     for part in re.split(r"[/,;|]+|\s+-\s+", options)]
            named = [part for part in parts if any(set(form.split()) <= part for form in wanted)]
            raw = [part.strip() for part in re.split(r"[/,;|]+|\s+-\s+", options)]
            if named and all(self.refused(requirement, text) for text, part in zip(raw, parts) if part in named):
                return -1
            if named:
                alone = any(not ((part & words) - value_words) and not Text.other_words(part - value_words) for part in named)
                mixed = named_as != "only" and self.mixed_colour(key, requirement)
                return 3 if alone and not mixed and named_as not in ("among", "other") else 1
            joined = [{Text.singular(w) for reading in (Text.lowered(part), Text.closed_up(part)) for w in re.split(r"[^a-z0-9.\-]+", reading) if w}
                      for part in re.split(r"[/,;|]+|\s+-\s+", options)]
            if any(set(form.split()) <= part for form in wanted for part in joined):
                return 1
            if stated:
                return 1 if options.strip() else 3
            if any(form.replace(" ", "") in Text.lowered(options).replace(" ", "") for form in wanted):
                return -1
            before = self.order.took_back.get(Text.lowered(requirement["value"]))
            if before and any(set(form.split()) <= part for form in self.forms(dict(requirement, value=before, written=[], alike=[])) for part in parts):
                return -1
            alike = any(set(form.split()) <= part for form in self.forms(requirement, loose=True) for part in parts)
            if alike or named_as in ("only", "among"):
                return 1
            if options.strip():
                return -1
            return 1 if listed else 0
        if kind == "material":
            told = self.options_material(options, requirement)
            if told is not None:
                return told
            exactly = any(self.entry_is(entry, requirement, label) and not re.search(r"[+&]", str(raw))
                          for (label, entry), raw in zip(singles, [value for label, value in self.named_facts(page, requirement)
                                                                 if len(Text.entries(value)) == 1]))
            if stated and exactly:
                return 3
            if stated:
                return 1
            if named_as in ("among", "other"):
                if named_as == "other" and self.says(title, requirement):
                    return 0
                return 1 if named_as == "among" else -1
            contrary = bool(singles)
            if named_as is None and self.says(options, requirement):
                named_too = ({Text.singular(w) for w in Text.word_list(options)} & words) - value_words
                return 1 if named_too else 2
            if named_as == "only":
                return 1 if contrary else 2
            if self.says(title, requirement):
                named_too = ({Text.singular(w) for w in Text.word_list(title)} & words) - value_words
                return 1 if named_too or contrary else 2
            if listed or self.says(title, requirement, loose=True):
                return 1
            if contrary:
                return -1
            return 0
        if stated:
            return 3 if any(self.entry_is(entry, requirement, label) for label, entry in singles) else 1
        if listed or any(self.entry_says(entry, requirement, label, loose=True) for label, entry in singles + lists):
            return 1
        if own:
            if self.filed_alike(page, requirement):
                return 1
            return -1
        if self.filed_elsewhere(page, requirement):
            return 1
        if related:
            return -1
        return 1 if elsewhere or self.says(title, requirement) else 0
    def exact(self, key: Key, requirement: Dict[str, Any]) -> int:
        return self.remembered("exact", key, requirement, lambda: self._exact(key, requirement))
    def _exact(self, key: Key, requirement: Dict[str, Any]) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        options = str(self.catalog.variant_of(key).get("options") or "")
        score = 0
        for label, value in self.named_facts(page, requirement):
            entries = Text.entries(value)
            if len(entries) == 1:
                score += 1 if self.entry_is(entries[0], requirement, label) else (0 if self.entry_says(entries[0], requirement, label) else -1)
        return score + int(self.states(options, requirement))
    def filed_elsewhere(self, page: Dict[str, Any], requirement: Dict[str, Any]) -> bool:
        mine = {label for label, _ in self.named_facts(page, requirement)}
        for label, value in (page.get("facts") or {}).items():
            entries = Text.entries(value)
            if str(label) not in mine and len(entries) == 1 and self.entry_is(entries[0], requirement, str(label)):
                return True
        return False
    def filed_alike(self, page: Dict[str, Any], requirement: Dict[str, Any]) -> bool:
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = Text.name_words(requirement["property"]) - {"type"}
        for label, value in (page.get("facts") or {}).items():
            entries = Text.entries(value)
            if (str(label) not in mine and wanted & Text.name_words(label) and len(entries) == 1
                    and self.entry_is(entries[0], requirement, str(label))):
                return True
        return False
    def label_nearness(self, label: str, requirement: Dict[str, Any]) -> Tuple[int, int, int]:
        wanted, have = Text.name_words(requirement["property"]), Text.name_words(label)
        shelf = Text.name_words(self.catalog.named_shelf() or self.order.category or self.order.item)
        other = have - wanted
        departments = bool(other) and other <= shelf
        return (len(wanted & have) - (0 if departments else len(other)), int(departments), len(wanted & have))
    def nearest_label(self, key: Key, requirement: Dict[str, Any]) -> int:
        return self.remembered("nearest_label", key, requirement, lambda: self._nearest_label(key, requirement))
    def _nearest_label(self, key: Key, requirement: Dict[str, Any]) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        mine = {label for label, _ in self.named_facts(page, requirement)}
        wanted = Text.name_words(requirement["property"])
        agree: Optional[Tuple[int, int, int]] = None
        differ: Optional[Tuple[int, int, int]] = None
        agree_own: Optional[Tuple[int, int, int]] = None
        differ_own: Optional[Tuple[int, int, int]] = None
        for label, value in (page.get("facts") or {}).items():
            label, entries = str(label), Text.entries(value)
            if not entries:
                continue
            is_value = len(entries) == 1 and self.entry_is(entries[0], requirement, label)
            if not is_value and label not in mine and len(wanted & Text.name_words(label)) < 2:
                continue
            near = self.label_nearness(label, requirement)
            if not is_value and len(entries) == 1 and self.entry_says(entries[0], requirement, label):
                is_value = True
            if not is_value and len(entries) > 1 and label in mine and any(self.entry_is(e, requirement, label) for e in entries):
                is_value = True
            if is_value:
                agree = near if agree is None else max(agree, near)
                if label in mine:
                    agree_own = near if agree_own is None else max(agree_own, near)
            else:
                differ = near if differ is None else max(differ, near)
                if label in mine:
                    differ_own = near if differ_own is None else max(differ_own, near)
        if agree is None:
            return -1 if differ is not None else 0
        if differ is None:
            return 1
        if agree == differ and agree_own == agree and (differ_own is None or differ_own < differ):
            return 1
        return (agree > differ) - (differ > agree)
    def usual_entry(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.remembered("usual_entry", key, requirement, lambda: self._usual_entry(key, requirement))
    def _usual_entry(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        longer = [entries[0] for label, value in self.named_facts(page, requirement) for entries in [Text.entries(value)]
                  if len(entries) == 1 and self.entry_says(entries[0], requirement, label) and not self.entry_is(entries[0], requirement, label)]
        if not longer:
            return True
        related, inferred, _ = self.own_labels(requirement)
        shelf = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        holding: Dict[str, int] = {}
        alone = 0
        for other in shelf:
            seen: Set[str] = set()
            for label, value in (other.get("facts") or {}).items():
                if str(label) not in related and str(label) not in inferred:
                    continue
                for entry in Text.entries(value):
                    if entry in seen:
                        continue
                    seen.add(entry)
                    if self.entry_is(entry, requirement, str(label)):
                        alone += 1
                    elif self.entry_says(entry, requirement, str(label)):
                        holding[entry] = holding.get(entry, 0) + 1
        most = max(holding.values(), default=0)
        return any(holding.get(entry, 0) == most and most >= max(alone, 1) for entry in longer)
    def listed_among(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        named = self.named_facts(page, requirement)
        alone = any(len(Text.entries(value)) == 1 and self.entry_says(Text.entries(value)[0], requirement, label) for label, value in named)
        among = any(len(Text.entries(value)) > 1 and any(self.entry_is(e, requirement, label) for e in Text.entries(value)) for label, value in named)
        return among and not alone
    def list_length(self, key: Key, requirement: Dict[str, Any]) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        lengths = [len(Text.entries(value)) for label, value in self.named_facts(page, requirement)
                   if len(Text.entries(value)) > 1 and any(self.entry_is(e, requirement, label) for e in Text.entries(value))]
        return min(lengths) if lengths else 0
    def options_name_other(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.remembered("options_name_other", key, requirement, lambda: self._options_name_other(key, requirement))
    def _options_name_other(self, key: Key, requirement: Dict[str, Any]) -> bool:
        mine = self.forms(requirement) | self.forms(requirement, loose=True)
        wanted = {w for form in mine for w in form.split()}
        related, inferred, _ = self.own_labels(requirement)
        known: Set[str] = set()
        for page in self.catalog.pages.values():
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    known |= {entry for entry in Text.entries(value) if re.fullmatch(r"[a-z]{3,}", entry)}
        known -= wanted
        options = str(self.catalog.variant_of(key).get("options") or "")
        parts = [{Text.singular(w) for w in re.split(r"[^a-z0-9]+", Text.lowered(part)) if w} for part in re.split(r"[/,;|]+", options)]
        named = any(set(form.split()) <= words for form in mine for words in parts)
        return not named and any(words & known for words in parts)
    def gainsaid(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.remembered("gainsaid", key, requirement, lambda: self._gainsaid(key, requirement))
    def _gainsaid(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        for label, value in self.named_facts(page, requirement):
            entries = Text.entries(value)
            if len(entries) == 1 and not self.entry_says(entries[0], requirement, label):
                return True
        return False
    def filed_materials(self, requirement: Dict[str, Any]) -> Set[str]:
        related, inferred, _ = self.own_labels(requirement)
        named = Text.name_words(self.order.item) | Text.name_words(self.catalog.named_shelf() or self.order.category)
        seen: Dict[str, Set[str]] = {}
        for pid, page in self.catalog.pages.items():
            for label, value in (page.get("facts") or {}).items():
                entries = Text.entries(value)
                if (str(label) in related or str(label) in inferred) and len(entries) == 1 and re.fullmatch(r"[a-z ]{3,}", entries[0]):
                    seen.setdefault(entries[0], set()).add(str(pid))
        return {entry for entry, pids in seen.items() if len(pids) >= 2 and not set(entry.split()) & named}
    def options_material(self, options: str, requirement: Dict[str, Any]) -> Optional[int]:
        words = [Text.singular(w) for w in re.split(r"[^a-z0-9]+", Text.lowered(options)) if w]
        if not words:
            return None
        text = " " + " ".join(words) + " "
        wanted = self.forms(requirement)
        value_words = {w for form in wanted for w in form.split()}
        if not any(" %s " % form in text for form in wanted):
            return None
        others = {m for m in self.filed_materials(requirement) - wanted if " %s " % m in text and not value_words & set(m.split())}
        return -1 if others else 3
    def level_needed(self, requirement: Dict[str, Any]) -> int:
        return 2 if self.kind(requirement) in ("amount", "material") else 3
    def settled(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.plain(key, requirement) >= 1
    def shade_filed(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        _, _, words = self.own_labels(requirement)
        value_words = {w for form in self.forms(requirement) for w in form.split()}
        for label, value in self.named_facts(page, requirement):
            entries = Text.entries(value)
            if len(entries) == 1 and self.entry_says(entries[0], requirement, label):
                beside = set(entries[0].split()) - value_words
                return bool(beside) and not (beside & words) and all(re.fullmatch(r"[a-z]{3,}", w) for w in beside)
        return False
    def second_colour(self, key: Key, requirement: Dict[str, Any]) -> bool:
        wanted = {w for form in self.forms(requirement) for w in form.split()}
        related, inferred, _ = self.own_labels(requirement)
        known: Set[str] = set()
        for page in self.catalog.pages.values():
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    known |= {entry for entry in Text.entries(value) if re.fullmatch(r"[a-z]{3,}", entry)}
        known -= wanted
        options = str(self.catalog.variant_of(key).get("options") or "")
        parts = [{Text.singular(w) for w in re.split(r"[^a-z0-9]+", Text.lowered(part)) if w} for part in re.split(r"[/,;|]+", options)]
        named = any(set(form.split()) <= words for form in self.forms(requirement) for words in parts)
        return named and any(words & known for words in parts)
    def mixed_colour(self, key: Key, requirement: Dict[str, Any]) -> bool:
        _, _, words = self.own_labels(requirement)
        wanted = {w for form in self.forms(requirement) for w in form.split()}
        others = words - wanted - {w for form in self.forms(requirement, loose=True) for w in form.split()}
        options = str(self.catalog.variant_of(key).get("options") or "")
        for part in re.split(r"\s+/\s+|[,;|]+", Text.lowered(options)):
            tokens = [Text.singular(t) if t[0].isalnum() and t != "and" else "+" for t in re.findall(r"[a-z0-9]+|(?<=[a-z])/(?=[a-z])|[+&]", part)]
            for at, token in enumerate(tokens[1:-1], start=1):
                if token == "+" and {tokens[at - 1], tokens[at + 1]} & wanted and {tokens[at - 1], tokens[at + 1]} & others:
                    return True
        return False
    def other_named(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.remembered("other_named", key, requirement, lambda: self._other_named(key, requirement))
    def _other_named(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        options = str(self.catalog.variant_of(key).get("options") or "")
        if not options.strip() or self.kind(requirement) != "fact" or self.says(options, requirement) or self.says(options, requirement, loose=True):
            return False
        value_words = {w for form in self.forms(requirement, loose=True) for w in form.split()}
        others: Set[str] = set()
        for label, value in self.named_facts(page, requirement):
            entries = Text.entries(value)
            if len(entries) >= 2:
                for entry in entries:
                    if not self.entry_says(entry, requirement, label):
                        others |= {w for w in entry.split() if re.fullmatch(r"[a-z]{3,}", w)}
        others -= value_words | Text.name_words(self.order.item) | Text.name_words(requirement["property"])
        if not others:
            return False
        parts = {Text.singular(w) for w in re.split(r"[^a-z0-9]+", Text.lowered(options)) if w}
        return bool(parts & others)


class Sieve(Part):
    def __init__(self) -> None:
        self.gone: Set[Key] = set()
        self.figures: Dict[Key, Optional[float]] = {}
        self.figure_unit = ""
        self.counted_cache: Dict[Tuple[Any, ...], Tuple[bool, Optional[float]]] = {}
        self.spelled: Set[str] = set()
        self.counted_stamp: Tuple[Any, ...] = ()
        self.counted_as = ""
        self.spelled_at = -1
        self.figure_room = FIGURE_LIMIT
        self.better = ""
        self.lost_titles: Set[str] = set()
        self.last_ranked: List[Key] = []
        self.figure_words: Dict[Key, str] = {}
    def affordable(self, key: Key) -> bool:
        seen = self.catalog.stock.get(key) or {}
        price = seen.get("price")
        if seen.get("in_stock") is False or key in self.gone:
            return False
        return self.order.budget is None or not isinstance(price, float) or price <= self.order.budget + 1e-9
    def just_over(self, key: Key, reach: float = OVER_REACH) -> bool:
        seen = self.catalog.stock.get(key) or {}
        price = seen.get("price")
        if self.order.budget is None or not isinstance(price, float) or seen.get("in_stock") is False or key in self.gone:
            return False
        first, opening = seen.get("first"), self.order.opening_budget
        if isinstance(first, float) and price > first + 1e-9 and opening is not None and price > opening + 1e-9:
            return False
        return self.order.budget + 1e-9 < price <= self.order.budget * reach
    def only_over(self) -> List[Key]:
        if self.order.moved_on:
            return []
        any_shelved = self.catalog.any_shelved()
        out = []
        for key in self.judge.verdicts:
            page = self.catalog.pages.get(key[0]) or {}
            verdict = self.judge.verdicts.get(key) or {}
            met = (verdict.get("met") or [])[: len(self.order.requirements)]
            fits_rest = bool(verdict.get("kind_ok")) and len(met) >= len(self.order.requirements) and all(x for x, r in zip(met, self.order.requirements) if not r.get("chosen"))
            if not self.just_over(key) or not fits_rest or (any_shelved and not self.catalog.in_category(page)):
                continue
            if self.order.brand and not self.order.brand_chosen and not set(Text.word_list(self.order.brand)) <= set(Text.word_list(page.get("brand"))):
                continue
            out.append(key)
        return sorted(out, key=lambda k: (-self.trust(k), (self.catalog.stock.get(k) or {}).get("price") or 0.0))
    def candidates(self) -> List[Key]:
        out = []
        any_shelved = self.catalog.any_shelved()
        for product_id, page in self.catalog.pages.items():
            if any_shelved and not self.catalog.in_category(page):
                continue
            for variant in page.get("variants") or []:
                key = (product_id, str(variant.get("sku")))
                if (self.affordable(key) or self.just_over(key)) and (not self.order.listed or key in self.order.listed):
                    out.append(key)
        return out
    def figure_on_page(self, key: Key, requirement: Dict[str, Any]) -> bool:
        figures = set(re.findall(r"\d+(?:\.\d+)?", requirement["value"]))
        if len(figures) < 2:
            return False
        page = self.catalog.pages.get(key[0]) or {}
        texts = [page.get("title"), self.catalog.variant_of(key).get("options")] + list((page.get("facts") or {}).values())
        return any(f in figures for text in texts for f in re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", Text.lowered(text)))
    def fits_words(self, key: Key) -> bool:
        verdict = self.judge.verdicts.get(key) or {}
        met = verdict.get("met") or []
        return bool(verdict.get("kind_ok")) and len(met) >= len(self.order.requirements) and all(met[: len(self.order.requirements)])
    def fits(self, key: Key) -> bool:
        if not self.fits_words(key) or not self.affordable(key):
            return False
        page = self.catalog.pages.get(key[0]) or {}
        shelved = self.catalog.in_category(page) or not self.catalog.any_shelved()
        brand = not self.order.brand or set(Text.word_list(self.order.brand)) <= set(Text.word_list(page.get("brand")))
        return shelved and brand
    def ranked(self) -> List[Key]:
        if self.clock.overdue() and self.last_ranked:
            return [k for k in self.last_ranked if self.fits(k)]
        self.last_ranked = self.ranking()
        return list(self.last_ranked)
    def ranking(self) -> List[Key]:
        if self.better in ("higher", "lower"):
            higher = self.better == "higher"
        else:
            higher = not re.search(r"\b(low|least|small|short|few|min|light)", Text.lowered(self.order.priority))
        by_price = self.priced_priority()
        by_figure = bool(self.order.priority) and not by_price
        fitting = [k for k in self.judge.verdicts if self.fits(k)]
        def number(key: Key) -> Optional[float]:
            value = self.figure(key)
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        figures = {k: number(k) for k in fitting} if by_figure else {}
        trust = {k: self.trust(k) for k in fitting}
        fixed = self.fixed_figure() if by_figure else None
        if fixed is not None and any(v is not None and abs(v - fixed) < 1e-9 for v in figures.values()):
            for key in fitting:
                if figures[key] is not None and abs(figures[key] - fixed) >= 1e-9:
                    trust[key] = round(trust[key] * FAINT_FIT, 4)
        measured = by_figure and bool(self.counted_unit()) and not self.counted_word()
        weight = {k: (1.0 if self.figure_stated(k, figures[k]) else SURE_FIT if measured and self.unit_unwritten(k) else UNSURE_FIT)
                  for k in fitting if figures.get(k) is not None}
        named: Dict[str, Tuple[float, float]] = {}
        top = max(trust.values(), default=0.0) or 1.0
        for k in weight:
            title = Text.lowered((self.catalog.pages.get(k[0]) or {}).get("title")) or k[0]
            value = figures[k] if higher else -figures[k]
            chance = min(1.0, trust[k] / top) * weight[k]
            if title not in named or (value, chance) > named[title]:
                named[title] = (value, chance)
        rivals = sorted(named.items(), key=lambda item: -item[1][0])
        sure = {k for k in fitting if trust[k] >= self.trust(k) - 1e-9 and all(self.reliance(k, r) >= SURE_FIT - 1e-9 for r in self.order.requirements)}
        def standing(key: Key) -> float:
            own = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title")) or key[0]
            mine = None if figures[key] is None else (figures[key] if higher else -figures[key])
            expected, left = 0.0, 1.0
            for title, (value, chance) in rivals:
                if title == own or (mine is not None and value <= mine):
                    continue
                if mine is not None and chance > 0:
                    share = (mine / value) if higher else (value / mine)
                    expected += left * chance * max(0.0, min(1.0, share))
                left *= 1.0 - chance
            if mine is None:
                return left
            none = 1.0
            for title, (_, chance) in rivals:
                if title != own:
                    none *= 1.0 - chance
            return (expected + left) * weight[key] + (1.0 - weight[key]) * none
        prices = sorted((self.catalog.stock.get(k) or {}).get("price") or 0.0 for k in fitting)
        middle = prices[len(prices) // 2] if prices else 0.0
        stated_any = any(figures.get(k) is not None for k in fitting)
        def order(key: Key) -> Tuple[Any, ...]:
            price = (self.catalog.stock.get(key) or {}).get("price") or 0.0
            plainly = sum(self.matcher.plain(key, r) for r in self.order.requirements) + sum(self.matcher.exact(key, r) for r in self.order.requirements)
            title = (self.catalog.pages.get(key[0]) or {}).get("title")
            twin = Text.lowered(title) in self.lost_titles
            named = sum(1 for r in self.order.requirements if self.matcher.says(title, r))
            rest = (-named, -self.hits(key), 0 if self.firm(key) else 1, 0 if self.solid(key) else 1, 0 if self.judge.verdicts[key].get("by") == "words" else 1, -plainly,
                    0 if twin else 1, abs(price - middle), price)
            if by_price:
                return (0, -trust[key], price) + rest
            if not by_figure:
                return (0, -trust[key]) + rest
            value = figures[key]
            worth = trust[key] * standing(key)
            in_options = value is not None and any(abs(float(n) - value) < 1e-9 for n in re.findall(r"\d+(?:\.\d+)?", Text.lowered(self.catalog.variant_of(key).get("options"))))
            ahead = key in sure and (value is not None or not stated_any)
            return (0, 0 if ahead else 1, -round(worth, 6), -trust[key], 1 if value is None else 0,
                    -((value if higher else -value) if value is not None else 0.0), 0 if in_options else 1) + rest
        result = sorted(fitting, key=order)
        return result
    def counted_unit(self) -> str:
        return self.counted_word() or Text.unit_word((re.findall(r"[a-z]+", Text.lowered(self.figure_unit)) or [""])[0])
    def counted_word(self) -> str:
        self.unit_spelled("")
        stamp = (self.order.priority, self.figure_unit, self.spelled_at)
        if self.counted_stamp != stamp:
            named = Text.unit_word((re.findall(r"[a-z]+", Text.lowered(self.figure_unit)) or [""])[0])
            self.counted_stamp, self.counted_as = stamp, ""
            for plural in Text.word_list(Text.lowered(self.order.priority)):
                word = Text.unit_word(plural)
                if (word != plural and len(word) >= 3 and not re.search(r"\d", word) and (not named or Text.same_unit(word, named))
                        and any(len(spelt) >= 2 and Text.same_unit(word, spelt) for spelt in self.spelled)):
                    self.counted_as = word
                    break
        return self.counted_as
    def unit_after(self, word: str, unit: str) -> Optional[float]:
        word = Text.unit_word(word)
        if len(unit) >= 2 and word.startswith(unit) and word != unit and not self.unit_spelled(word):
            return None
        return 1.0 if Text.same_unit(word, unit) else None
    def unit_spelled(self, word: str) -> bool:
        if self.spelled_at != self.catalog.page_writes:
            seen: Dict[str, Set[str]] = {}
            for page in self.catalog.pages.values():
                texts = ([page.get("title")] + list((page.get("facts") or {}).values())
                         + [v.get("options") for v in page.get("variants") or [] if isinstance(v, dict)])
                for text in texts:
                    for m in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?)\s?([a-z]+|\")(?![a-z0-9])", Text.lowered(text)):
                        seen.setdefault(Text.unit_word(m.group(2)), set()).add(m.group(1))
            spelled = {w for w, figures in seen.items() if len(figures) >= 2}
            if spelled != self.spelled:
                self.counted_cache.clear()
            self.spelled, self.spelled_at = spelled, self.catalog.page_writes
        return word in self.spelled
    def label_counts(self, label: Any, unit: str) -> Optional[float]:
        words = Text.word_list(str(label).replace("_", " "))
        if not {Text.singular(w) for w in words} & Text.content_words(self.order.priority):
            return None
        return 1.0 if any(Text.unit_word(word) == unit for word in words) else None
    def counted_figure(self, key: Key) -> Tuple[bool, Optional[float]]:
        unit = self.counted_unit()
        if not unit:
            return (False, None)
        stamp = (unit, key)
        if stamp in self.counted_cache:
            return self.counted_cache[stamp]
        page = self.catalog.pages.get(key[0]) or {}
        def written(text: Any) -> Set[float]:
            out: Set[float] = set()
            for m in re.finditer(r"(?<![\w.,])(\d+(?:\.\d+)?)\s?([a-z]+|\")(?![a-z0-9])", Text.lowered(text)):
                step = self.unit_after(m.group(2), unit)
                if step:
                    out.add(round(float(m.group(1)) * step, 6))
            return out
        facts: Set[float] = set()
        for label, fact in (page.get("facts") or {}).items():
            facts |= written(fact)
            bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
            step = self.label_counts(label, unit) if bare else None
            if bare and step:
                facts.add(round(float(bare.group(1)) * step, 6))
        answer: Tuple[bool, Optional[float]] = (False, None)
        for found in (written(self.catalog.variant_of(key).get("options")), facts, written(page.get("title"))):
            if found:
                answer = (True, next(iter(found)) if len(found) == 1 else None)
                break
        else:
            if self.counted_word() or self.unit_uncounted(key, unit):
                answer = (True, None)
        self.counted_cache[stamp] = answer
        return answer
    def unit_uncounted(self, key: Key, unit: str) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        texts = [self.catalog.variant_of(key).get("options"), page.get("title")] + list((page.get("facts") or {}).values())
        return any(self.unit_after(m.group(2), unit) for text in texts if text
                   for m in re.finditer(r"(\d+(?:\.\d+)?)\s?([a-z]+|\")(?![a-z0-9])", Text.lowered(text)))
    def figure_stated(self, key: Key, value: float) -> bool:
        words = self.figure_words.get(key)
        if words and Text.says_amount(words, float(value)):
            page = self.catalog.pages.get(key[0]) or {}
            texts = [self.catalog.variant_of(key).get("options"), page.get("title")] + list((page.get("facts") or {}).values())
            if any(Text.closed_up(words) in Text.closed_up(text) for text in texts if text):
                return True
        unit = self.counted_unit()
        if not unit:
            return True
        settled, counted = self.counted_figure(key)
        if settled:
            return counted is not None and abs(counted - float(value)) < 1e-9
        def alike(found: str) -> bool:
            return Text.same_unit(found, unit)
        def stands_in(text: Any) -> Optional[bool]:
            found = [(float(m.group(1)), m.group(2)) for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+|\")(?![a-z0-9])", Text.lowered(text))]
            same = {number for number, word in found if alike(word)}
            if same:
                return same == {float(value)}
            return None
        options, title = self.catalog.variant_of(key).get("options"), (self.catalog.pages.get(key[0]) or {}).get("title")
        for text in (options, title):
            stated = stands_in(text)
            if stated is not None:
                return stated
        for label, fact in ((self.catalog.pages.get(key[0]) or {}).get("facts") or {}).items():
            if stands_in(fact) is True:
                return True
            bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", str(fact))
            if bare and float(bare.group(1)) == float(value) and any(Text.same_unit(word, unit) for word in Text.word_list(str(label).replace("_", " "))):
                return True
        return False
    def unit_unwritten(self, key: Key) -> bool:
        unit = self.counted_unit()
        page = self.catalog.pages.get(key[0]) or {}
        if not unit or not page:
            return False
        texts = [self.catalog.variant_of(key).get("options"), page.get("title")] + list((page.get("facts") or {}).values())
        for text in texts:
            for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s?([a-z]+|\")(?![a-z0-9])", Text.lowered(text)):
                word = Text.unit_word(m.group(2))
                if Text.same_unit(word, unit):
                    return False
        return True
    def fixed_figure(self) -> Optional[float]:
        unit = Text.unit_word((re.findall(r"[a-z]+", Text.lowered(self.figure_unit)) or [""])[0])
        if not unit:
            return None
        for requirement in self.order.requirements:
            for form in self.matcher.forms(requirement):
                for found in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", form):
                    if Text.unit_word(found.group(2)) == unit:
                        return float(found.group(1))
        return None
    def trust(self, key: Key) -> float:
        worth = 1.0
        for requirement in self.order.requirements:
            worth *= self.reliance(key, requirement)
        return round(worth, 4)
    def reliance(self, key: Key, requirement: Dict[str, Any]) -> float:
        return self.matcher.remembered("reliance", key, requirement, lambda: self.read_alike(key, requirement, self._reliance(key, requirement)))
    def lead_words(self, key: Key) -> str:
        parts = [p for p in re.split(r"[/,;|+]+", Text.lowered(self.catalog.variant_of(key).get("options") or "")) if p.strip()]
        return " ".join(Text.word_list(parts[0])) if parts else ""
    def read_alike(self, key: Key, requirement: Dict[str, Any], value: float) -> float:
        if value >= SURE_FIT or self.matcher.kind(requirement) != "colour":
            return value
        if self.matcher.named_as.get((key, requirement["property"], requirement["value"])) not in (None, "none", "unread"):
            return value
        word = self.lead_words(key)
        if not word or re.search(r"\d", word) or set(word.split()) & set(Text.word_list(Text.lowered(requirement["value"]))):
            return value
        for (other, prop, wanted), mark in self.matcher.named_as.items():
            if mark == "only" and other != key and prop == requirement["property"] and wanted == requirement["value"] and self.lead_words(other) == word:
                return SURE_FIT
        return value
    def _reliance(self, key: Key, requirement: Dict[str, Any]) -> float:
        level, kind = self.matcher.plain(key, requirement), self.matcher.kind(requirement)
        if kind == "fact":
            page = self.catalog.pages.get(key[0]) or {}
            alone = level >= 3 and any(len(Text.entries(value)) == 1 and self.matcher.entry_is(Text.entries(value)[0], requirement, label)
                                       for label, value in self.matcher.named_facts(page, requirement))
            if level >= 1 and not alone and self.matcher.other_named(key, requirement):
                return UNSURE_FIT
            nearest = self.matcher.nearest_label(key, requirement)
            listed = self.matcher.list_length(key, requirement) if self.matcher.listed_among(key, requirement) else 0
            if level >= 3:
                if nearest < 0:
                    return FAINT_FIT
                if nearest > 0 and listed:
                    return self.listed_fit(listed)
                if nearest > 0 and self.matcher.exact(key, requirement) > 0 and not self.matcher.gainsaid(key, requirement):
                    return FIRM_FIT
                return SURE_FIT if nearest > 0 else LIKELY_FIT
            if level >= 1 and nearest < 0:
                return FAINT_FIT
            if level >= 1:
                if listed:
                    return self.listed_fit(listed)
                within = any(len(Text.entries(value)) == 1 and self.matcher.entry_says(Text.entries(value)[0], requirement, label)
                             or any(self.matcher.entry_is(e, requirement, label) for e in Text.entries(value))
                             for label, value in self.matcher.named_facts(page, requirement))
                related, inferred, _ = self.matcher.own_labels(requirement)
                if (related or inferred) and not self.matcher.named_facts(page, requirement):
                    return UNSURE_FIT
                if within and not listed and not self.matcher.usual_entry(key, requirement):
                    return UNSURE_FIT
                if within:
                    return SURE_FIT
                if self.matcher.filed_alike(page, requirement):
                    return LIKELY_FIT
                if self.matcher.filed_elsewhere(page, requirement):
                    return UNSURE_FIT
            return FAINT_FIT
        if kind == "amount":
            if level == 1 and (self.matcher.converted_on(key, requirement) or self.matcher.last_side(key, requirement)):
                return LIKELY_FIT
            return FIRM_FIT if level >= 2 else UNSURE_FIT if level == 1 else FAINT_FIT
        if kind == "colour":
            named_as = self.matcher.named_as.get((key, requirement["property"], requirement["value"]))
            if named_as in ("only", "among") and level >= 1 and str(self.catalog.variant_of(key).get("options") or "").strip():
                if named_as == "among":
                    return UNSURE_FIT
                if level < 3:
                    return SURE_FIT
            if self.matcher.options_name_other(key, requirement):
                return FAINT_FIT
            if level >= 1 and named_as != "only" and self.matcher.mixed_colour(key, requirement):
                return UNSURE_FIT
            if level >= 3:
                own_word = dict(requirement, written=[], alike=[])
                if self.matcher.second_colour(key, requirement):
                    return UNSURE_FIT
                return FIRM_FIT if self.matcher.states(self.catalog.variant_of(key).get("options"), own_word) or self.matcher.exact(key, requirement) > 0 else SURE_FIT
            own = (self.matcher.says(self.catalog.variant_of(key).get("options"), requirement) or self.matcher.exact(key, requirement) > 0
                   or self.matcher.shade_filed(key, requirement))
            if level >= 1 and own and self.matcher.named_as.get((key, requirement["property"], requirement["value"])) in ("among", "other"):
                return UNSURE_FIT
            if level >= 1 and own and self.matcher.second_colour(key, requirement):
                return UNSURE_FIT
            return SURE_FIT if level >= 1 and own else UNSURE_FIT if level >= 1 else FAINT_FIT
        if level >= 3:
            return FIRM_FIT
        if level == 2:
            return SURE_FIT
        if level == 1:
            return UNSURE_FIT if self.matcher.list_length(key, requirement) > 3 else SURE_FIT
        return FAINT_FIT
    @staticmethod
    def listed_fit(length: int) -> float:
        return LISTED_SHORT if length <= 3 else LISTED_LONG if length <= 7 else FAINT_FIT
    def keeps_request(self, key: Key) -> bool:
        stated = [r for r in self.order.requirements if r.get("stated")]
        return bool((self.judge.verdicts.get(key) or {}).get("kind_ok")) and all(self.matcher.plain(key, r) >= self.matcher.level_needed(r) for r in stated)
    def doubts(self, key: Key) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        title, options = page.get("title"), self.catalog.variant_of(key).get("options")
        count = 0
        for requirement in self.order.opening:
            kind = self.matcher.kind(requirement)
            if self.matcher.gainsaid(key, requirement) or self.matcher.nearest_label(key, requirement) < 0:
                count += 1
            elif any(len(Text.entries(value)) > 1 for _, value in self.matcher.named_facts(page, requirement)):
                count += 1
            if kind == "amount":
                stated, others = self.matcher.amounts(options, requirement)
                if others or (self.matcher.amounts(title, requirement)[1] and not stated):
                    count += 1
            if kind == "colour" and self.matcher.second_colour(key, requirement):
                count += 1
            if kind in ("material", "fact"):
                _, _, known = self.matcher.own_labels(requirement)
                mine = {w for form in self.matcher.forms(requirement) for w in form.split()}
                if ({Text.singular(w) for w in Text.word_list(title)} & known) - mine:
                    count += 1
        return count
    def own_exact(self, key: Key) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        count = 0
        for requirement in self.order.opening:
            written, wanted = " ".join(Text.word_list(requirement["property"])), Text.name_words(requirement["property"])
            for label, value in (page.get("facts") or {}).items():
                if " ".join(Text.word_list(str(label).replace("_", " "))) != written and Text.name_words(label) != wanted:
                    continue
                entries = Text.entries(value)
                if len(entries) == 1 and self.matcher.entry_is(entries[0], requirement, str(label)):
                    count += 1
                    break
        return count
    def amount_written(self, key: Key, requirement: Dict[str, Any]) -> bool:
        wanted = {(float(m.group(1)), Text.unit_word(m.group(2))) for form in self.matcher.forms(requirement)
                  for m in re.finditer(r"(\d+(?:\.\d+)?)\s*([a-z]+|\")", Text.lowered(form))}
        if not wanted:
            return True
        page = self.catalog.pages.get(key[0]) or {}
        texts = [self.catalog.variant_of(key).get("options"), page.get("title")] + list((page.get("facts") or {}).values())
        for text in texts:
            for m in re.finditer(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)\s*([a-z]+|\")", Text.lowered(text)):
                if (float(m.group(1)), Text.unit_word(m.group(2))) in wanted:
                    return True
        return False
    def meets_opening(self, key: Key) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        seen = self.catalog.stock.get(key) or {}
        price = seen.get("price")
        if key in self.gone or seen.get("in_stock") is False or not isinstance(price, float):
            return False
        if any(limit is not None and price > limit + 1e-9 for limit in (self.order.opening_budget, self.order.budget)):
            return False
        if not (self.judge.verdicts.get(key) or {}).get("kind_ok"):
            return False
        if not self.catalog.in_category(page) and self.catalog.any_shelved():
            return False
        if self.order.brand and not set(Text.word_list(self.order.brand)) <= set(Text.word_list(page.get("brand"))):
            return False
        return all(self.matcher.plain(key, r) >= self.matcher.level_needed(r) and (self.matcher.kind(r) != "amount" or self.amount_written(key, r))
                   for r in self.order.opening)
    def meets_chosen(self, key: Key) -> bool:
        met = ((self.judge.verdicts.get(key) or {}).get("met") or [])[: len(self.order.requirements)]
        marks = [x for x, r in zip(met, self.order.requirements) if r.get("chosen")]
        return bool(marks) and all(marks)
    def priced_priority(self) -> bool:
        text = Text.lowered(self.order.priority)
        paid = r"(?:price|cost|expensive|afford\w*|pay\w*|spend\w*)"
        return bool(re.search(r"\bcheap|\bbest (?:price|value|deal)\b|\bmost afford", text)
                    or re.search(rf"\b(?:low\w*|least|less|small\w*|minim\w*)\b(?:\s+\w+){{0,2}}?\s+{paid}", text)
                    or re.search(rf"\b{paid}\s+(?:\w+\s+)?(?:as\s+)?(?:low|little|less|least)\b", text))
    def figure(self, key: Key) -> Any:
        settled, counted = self.counted_figure(key) if key[0] in self.catalog.pages else (False, None)
        read = self.figures.get(key)
        if (isinstance(read, (int, float)) and not isinstance(read, bool) and float(read) != 1.0 and self.counted_word()
                and self.one_unit(key, float(read))):
            read = 1.0
        if settled and counted is not None and isinstance(read, (int, float)) and not isinstance(read, bool) and counted != read:
            in_options = {float(n) for n in re.findall(r"\d+(?:\.\d+)?", Text.lowered(self.catalog.variant_of(key).get("options")))}
            if float(read) in in_options and counted not in in_options:
                return read
        if settled:
            return counted
        if key in self.figures:
            return read
        return (self.judge.verdicts.get(key) or {}).get("rank_value")
    def one_unit(self, key: Key, value: float) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        texts = [self.figure_words.get(key), self.catalog.variant_of(key).get("options"), page.get("title")]
        number = f"{value:g}"
        in_one = rf"(?<![\d.]){re.escape(number)}\s*-?\s*(?:in|n)\s*-?\s*1(?![\d.])"
        counted = self.counted_word()
        after = [m.group(1) for t in texts if t for m in re.finditer(rf"(?<![\d.]){re.escape(number)}\s*-?\s*([a-z]+)", Text.lowered(t))]
        alone = bool(counted) and any(Text.same_unit(Text.unit_word(w), counted) for w in after)
        return any(re.search(in_one, Text.lowered(t)) for t in texts if t) and not alone
    def best_figures(self) -> List[Tuple[float, Key]]:
        lower = self.better == "lower"
        known = [(float(self.figure(k)), k) for k in self.ranked() if k not in self.gone
                 and isinstance(self.figure(k), (int, float)) and not isinstance(self.figure(k), bool)]
        return sorted(known, key=lambda pair: pair[0] if lower else -pair[0])
    def solid(self, key: Key) -> bool:
        return (self.judge.verdicts.get(key) or {}).get("by") == "words" and all(self.matcher.plain(key, r) >= 2 for r in self.order.requirements)
    def firm(self, key: Key) -> bool:
        if (self.judge.verdicts.get(key) or {}).get("by") != "words":
            return False
        return all(self.matcher.plain(key, r) >= self.matcher.level_needed(r) for r in self.order.requirements)
    def on_sale(self, key: Key) -> bool:
        seen = self.catalog.stock.get(key) or {}
        price = seen.get("price")
        return seen.get("in_stock") is not False and (self.order.budget is None or not isinstance(price, float) or price <= self.order.budget + 1e-9)
    def value_hit(self, key: Key, requirement: Dict[str, Any]) -> bool:
        return self.matcher.remembered("value_hit", key, requirement, lambda: self._value_hit(key, requirement))
    def _value_hit(self, key: Key, requirement: Dict[str, Any]) -> bool:
        queries = self.catalog.found_by.get(key[0]) or set()
        if not queries:
            return False
        forms = [form.split() for form in self.matcher.forms(requirement) if form]
        for query in queries:
            words = Text.word_list(query)
            if any(all(any(Text.same_word(w, q) for q in words) for w in form) for form in forms):
                return True
        return False
    def hits(self, key: Key) -> int:
        return sum(1 for r in self.order.requirements if self.reliance(key, r) < FIRM_FIT and self.value_hit(key, r))
    def word_on_page(self, key: Key, requirement: Dict[str, Any]) -> bool:
        page = self.catalog.pages.get(key[0]) or {}
        texts = [page.get("title"), self.catalog.variant_of(key).get("options")] + list((page.get("facts") or {}).values())
        return any(self.matcher.says(text, requirement) for text in texts if text)
    def near_misses(self) -> List[Key]:
        def score(key: Key) -> Tuple[Any, ...]:
            verdict = self.judge.verdicts[key]
            met = (verdict.get("met") or [])[: len(self.order.requirements)]
            chosen = sum(1 for x, r in zip(met, self.order.requirements) if x and r.get("chosen"))
            carried = sum(1 for x, r in zip(met, self.order.requirements) if not x and self.word_on_page(key, r))
            found = sum(1 for x, r in zip(met, self.order.requirements) if not x and self.value_hit(key, r))
            against = sum(1 for x, r in zip(met, self.order.requirements) if not x and self.matcher.plain(key, r) < 0)
            surely = round(sum(self.reliance(key, r) for x, r in zip(met, self.order.requirements) if x), 3)
            figure = self.figure(key)
            standing = -float(figure) if isinstance(figure, (int, float)) and not isinstance(figure, bool) else 0.0
            if self.better == "lower":
                standing = -standing
            return (-int(bool(verdict.get("kind_ok"))), -chosen, -sum(1 for x in met if x), -carried, -found, against, -self.said_in_part(key, met),
                    -surely, standing, (self.catalog.stock.get(key) or {}).get("price") or 0.0)
        return sorted((k for k in self.judge.verdicts if self.affordable(k)), key=score)
    def closest(self) -> Optional[Key]:
        def nearness(key: Key) -> Tuple[Any, ...]:
            verdict = self.judge.verdicts.get(key) or {}
            met = (verdict.get("met") or [])[: len(self.order.requirements)]
            stated = sum(1 for x, r in zip(met, self.order.requirements) if x and r.get("stated"))
            carried = sum(1 for x, r in zip(met, self.order.requirements) if not x and self.word_on_page(key, r))
            found = sum(1 for x, r in zip(met, self.order.requirements) if not x and self.value_hit(key, r))
            against = sum(1 for x, r in zip(met, self.order.requirements) if not x and self.matcher.plain(key, r) < 0)
            return (-int(bool(verdict.get("kind_ok"))), -stated, -sum(1 for x in met if x), -carried, -found, against, -self.said_in_part(key, met),
                    -int(key in self.catalog.cart), (self.catalog.stock.get(key) or {}).get("price") or 0.0)
        open_keys = [k for k in self.judge.verdicts if self.on_sale(k)]
        return min(open_keys, key=nearness) if open_keys else None
    def said_in_part(self, key: Key, met: List[Any]) -> int:
        page = self.catalog.pages.get(key[0]) or {}
        count = 0
        for position, requirement in enumerate(self.order.requirements):
            if position < len(met) and met[position] or re.search(r"\d", requirement["value"]):
                continue
            wanted = {Text.singular(w) for form in self.matcher.forms(requirement) for w in form.split()}
            for label, value in self.matcher.named_facts(page, requirement):
                named = Text.name_words(label)
                if any(words and all(any(Text.same_word(w, v) for v in wanted) for w in words)
                       for words in ([w for w in entry.split() if not any(Text.same_word(w, n) for n in named)] for entry in Text.entries(value))):
                    count += 1
                    break
        return count


JUDGE_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in two or three sentences: what each requirement needs a listing's words to say, and which listings come closest and why."},
    "marks": Schema.strings("One short entry for every listing shown, in the order shown: the listing's number, then K when it is the kind of thing asked for or X when it is not, "
                     "then one letter per requirement in order - Y when the listing's own words state that value, N when they do not. Example for two requirements: '3 K YN'."),
    "proof": {"type": "array", "description": "Only for listings marked K with every requirement Y: where each requirement is stated.", "items": {
        "type": "object", "properties": {
            "n": {"type": "integer", "description": "The listing's number."},
            "where": Schema.strings("For each requirement in order: title, options, or the fact's label exactly as written."),
            "words": Schema.strings("For each requirement in order: the listing's exact words that state it."),
            "rank_value": {"type": "number", "description": "The listing's number for the shopper's priority when one is given, else 0."}},
        "required": ["n", "where", "words"]}},
    "queries": Schema.strings("When no listing meets every requirement: up to five other short searches worth trying; otherwise empty."),
    "notes": Schema.strings("Written last: one entry for every listing marked X or with an N, '<number>: ' and then, in a few words, what its own words say instead for the first requirement it misses (or that they say nothing about it)."),
}, ["reason", "marks", "notes"])

JUDGE_INSTRUCTIONS = """You check store listings against a shopper's requirements for a shopping assistant. The shopper trusts the assistant only if the item ordered meets every one of their requirements; one requirement read wrongly loses that trust. Read exactly what the words say; never assume, guess or fill in what they do not say.

WHAT YOU ARE GIVEN
The shopper's own words in order (the request and every line since): read them for what each requirement means. Then the item, the requirements in order (property: value), sometimes the priority, and numbered listings - each with its title, the options of the exact variant that would be ordered, its price, shelf, brand and facts, and after '||' the fact named for a requirement where the page has one.

WHAT YOU RETURN
- reason: written first, in two or three sentences: what each requirement needs a listing's words to say, and which listings come closest and why.
- marks: one entry for every listing, in the order shown: '<number> <K or X> <one letter per requirement>'. K when the listing is the kind of thing asked for, X when it is not; then for each requirement in order, Y when the listing's own words state that value for the variant that would be ordered, N when they do not. For two requirements: '<number> K YN'.
- proof: only for listings marked K with every requirement Y - for each requirement in order, where it is stated (title, options, or the fact's label exactly as written) and the listing's exact words that state it; rank_value is the listing's figure for the shopper's priority when one is given.
- queries: when no listing meets every requirement, up to five other short searches worth trying; otherwise empty.
- notes: written last - one entry for every listing marked X or with an N: '<number>: ' and, in a few words, what its own words say instead for the first requirement it misses (or that they say nothing of it).

HOW TO JUDGE ONE REQUIREMENT - take the first rule that applies.
1. The variant's options. The options describe the exact item sold. A colour, a size or an amount they name decides that property: when they name another colour or amount, the requirement is N whatever the title lists (a title often lists the colours or sizes of every variant).
2. The fact named for the requirement. When the listing shows the fact named for requirement N, that fact decides requirement N: Y when it states the value, N when it states something else, unless the options state the value. A fact that states the value settles it even when the title's words suggest otherwise.
3. Where each kind of value stands. A material or an amount is stated by the title, the options or a fact; any other property the store keeps as a fact of its own is stated by the fact whose label names that property.
4. The same value in other words. A requirement is met by the same word, a plain synonym or another spelling of it, or the same amount written in another unit: convert units yourself.
5. Amounts. An amount counts when it is one of the measures written for the variant and it is the measure the requirement asks for; a range that holds the amount counts. One figure of a longer measurement (length x width x height) states only the dimension it stands for. An amount filed under a fact for a different quantity, or the figure of another variant listed in the title, does not state it; nor does a figure of another measure of the item (an opening, a cut-out, a hole, a thread, a cord beside the item's own size), nor the figure of a part sold with it. A deal written as buy <n> take <m>, or its short form, is a deal, not a count of pieces. '<n> in 1' names one unit with <n> uses or parts: one piece, not <n>.
6. Colours. A light or dark shade of the colour states it; a name that joins it to another colour or to a metal does not.
7. Kinds. A narrower kind of the required value states it (a specific kind of the wanted material or type); a wider kind does not.
8. Something else. A value said only of a part, an accessory, a free gift, the packaging, another item of a set, or of what the item fits, does not state it for the item.
9. Another variant. When another variant of the listing names the required amount in its options and this variant's options name a different figure, the amount is that other variant's: N.
10. Absent is N. A value the listing's words do not state is unmet; never assume it.

HOW TO JUDGE THE KIND - K when the listing is the kind of thing the shopper's words ask for (another name for the same kind counts); X when it is another kind of thing, only a part or an accessory of it, or a set built around another item.

Mark every listing, in order."""

SEAT_JUDGE = ModelSeat('judge', 'Which listings state every requirement.', JUDGE_INSTRUCTIONS, JUDGE_SCHEMA, 8192, (GLM_FLASH, GLM_FLASH, GLM))

NAMES_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one or two sentences: which words of the texts name the property and how you told them apart."},
    "marks": {"type": "array", "items": {"type": "string"}, "description": "One entry per text, in order: '<number> <only|among|other|none>'."}}, ["reason", "marks"])

NAMES_INSTRUCTIONS = """You read, for a shopping assistant, what a listing's own words say about one property of an item: its colour, or the material it is made of. The shopper trusts the assistant only if the item ordered meets every one of their requirements; one read wrongly loses that trust. Read exactly what the words say; never assume, guess or fill in what they do not say.

WHAT YOU ARE GIVEN
The item, the property, the wanted value, and numbered texts: the variant's own words or the listing's title.

WHAT YOU RETURN
- reason: written first, in one or two sentences: which words of the texts name the property and how you told them apart.
- marks: one entry per text, in order, '<number> <only|among|other|none>':
  only - the words name the wanted value (the same word, or another word sellers use for that same value) and no other value of that property;
  among - the words name the wanted value together with another value of that property (two colours, two materials), so it is unclear which one the item is;
  other - the words name a value of that property, but not the wanted one;
  none - the words name no value of that property.

RULES
1. A light or dark shade or tint of the wanted colour (a word that only says how light, dark, pale, deep or bright it is: light-, dark-, off-) is the wanted colour.
2. Read the words part by part: '+', '/', ',', '|' and a spaced dash join separate parts (a colour, a print or picture, a shape, a size, a piece of the item). A part that names a print, a picture, a pattern, a shape, a size or a piece says nothing of the colour - that holds even when the thing it pictures has a colour of its own, and it holds before every rule below. Within one part, a single name that joins two different colours (the wanted colour beside a word that is itself a colour's name), or a colour and a metal, is another colour: other. The wanted colour's own word beside words that name no colour (a thing, a flavour, a scent, a model or a style) still names the wanted colour. A part that holds no colour word at all names no colour: none - a scent, a flavour, a fruit, a flower, a plant or a stone named on its own, whatever colour that thing usually has.
3. The variant's words describe the exact item sold: a material they name is the item's material, whatever the title lists.
3a. When a fact of the listing gives that same word as the item's pattern, print or design, the word in the variant's name pictures that design and names no colour: none.
4. Judge only the property asked about: sizes, codes, counts and other details are not values of it. A word cut short or misspelled (letters dropped, swapped or changed) names no value of the property: none.
5. A name that gives two materials of the item is among. A material named of a part of the item itself (its handle, its strap, its base) is the item's material: only. A material named only of what the item fits or goes with (a holder for a <material> thing) is none for the item.
Mark every text, in order."""

SEAT_NAMES = ModelSeat('names', 'What each text names for one property.', NAMES_INSTRUCTIONS, NAMES_SCHEMA, 4096, (GLM_FLASH, GLM_FLASH, GLM))

RANK_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one or two sentences: where these listings state the figure (options, facts or title), and how you wrote them all in one unit."},
    "values": {"type": "array", "items": {"type": "string"}, "description": "One entry per listing, in order: '<number> <figure> \"<words>\"' with the figure as a plain number in one unit for every listing and the listing's own words it was read from, in double quotes; or '<number> none' when the listing's words state none."},
    "better": {"type": "string", "enum": ["higher", "lower"], "description": "Whether a higher or a lower figure is better for this priority."},
    "unit": {"type": "string", "description": "The one unit every figure is written in (empty for a plain count)."}}, ["reason", "values", "better", "unit"])

RANK_INSTRUCTIONS = """You read store listings for a shopping assistant. A shopper said what makes one fitting item better than another: their priority. For each listing, read the figure this priority is about from the listing's own words, for the exact variant that would be ordered. The shopper trusts the assistant only if what it reads is what the page says: never assume, guess or fill in.

WHAT YOU ARE GIVEN
The item, the shopper's priority, sometimes the unit every figure must be written in, then numbered listings: the title, the variant's options, the price, the shelf, the brand and the facts.

WHAT YOU RETURN
- reason: written first, in one or two sentences: where these listings state the figure (options, facts or title), and how you wrote them all in one unit.
- values: one entry per listing, in order: '<number> <figure> "<words>"' - the figure as a plain number, then the listing's own words you read it from, exactly as written, in double quotes; or '<number> none' when the listing's words state no such figure.
- better: whether a higher or a lower figure is better for this priority.
- unit: the one unit every figure is written in (empty for a plain count).

WHERE TO READ THE FIGURE - in this order:
1. The variant's options. They describe the exact item sold: when they state the figure, that is the figure, whatever the title lists (a title often lists the amounts of all its variants).
2. A fact whose label names the measure.
3. The title, when it states one figure of that kind for the item.

WHAT COUNTS AS THE FIGURE
1. Write every figure in the same unit, converting where needed.
2. A count (the most of something countable) is the number of separate units one order of this variant brings: pieces, items in its pack or set, units in its box or bottle (the units a pack holds count when the priority counts such units). One unit, written as one piece, is the figure 1. A number that describes one unit rather than how many units come is not the count: its size, its model or series, a code, how many parts, positions, holes, cups, slots, compartments, functions, settings or layers one unit has. '<n> in 1', '<n>-in-1' or '<n>n1' names one unit with <n> uses or parts: the count is 1, and so does a set named by how many parts it has. A count is a number written with a word for pieces or packs; a bare number under a fact, with no such word beside it, is not a count: write none.
2a. The words next to a number say what it counts: a count stands beside a word for separate units (pieces, sets, packs, the item's own name in the plural); a number beside a measure's unit is an amount, not a count.
3. A deal (two for the price of one, a unit given free, a bonus) adds nothing: read only the count or the amount written for the variant itself.
4. A size written as several figures joined by signs: the last figure, the one the unit follows - never the first or the largest of them. Several figures joined by letters, or a range, are no single figure of the item: take the upper end of a range; write none for figures joined by letters.
4a. A size on a sizing scale - the size number or letter an item worn, fitted or slept on is sold in - is a size, not a measure: never write it as a length, width, height, volume or weight, and never as a count. A word that names a kind or a level rather than an amount is no figure either: write none.
4b. A fact whose whole value is two bare numbers joined by a dash, with no unit in the value or in its label, is a bracket the store's form offers for many items, not this item's figure: write none for it. A range written with its unit is read as rule 4 says.
5. Write none when the listing states no such figure; never guess one. Read every listing the same way: when one listing states no figure in a unit of the measure, none is its figure, whatever another listing states.
6. Several figures of the same kind in the title list the variants sold under it: take the one the variant's options name; when the options name none, write none.
7. A count of pieces is never the figure of a measure, and a measure is never a count: for a priority about an amount (how much the item holds, weighs, measures or draws) write none when the page states no such amount for the variant - never its number of pieces; for a priority about pieces write only a count of pieces.
One entry per listing."""

SEAT_RANK = ModelSeat('rank', "Each listing's figure for the shopper's priority.", RANK_INSTRUCTIONS, RANK_SCHEMA, 4096, (GLM_FLASH, GLM_FLASH, GLM))

DECIDE_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in two to four sentences: the requirements on the item that the shopper's words set as they stand now, which of the assistant's notes differ from those words, and how the candidates compare."},
    "requirements": {"type": "array", "description": "Every requirement on the item itself that holds now, one entry each, with every candidate sorted under it.", "items": {
        "type": "object", "properties": {
            "requirement": {"type": "string", "description": "'<property>: <value>' in the shopper's words. A property of the item itself only: never the price, the spending limit, the stock, the store's category, the shopper's priority between fitting items, or why or for whom the shopper buys."},
            "stated_by": {"type": "array", "items": {"type": "integer"}, "description": "The numbers of the candidates whose words state this value for the variant that would be ordered."},
            "missed_by": {"type": "array", "items": {"type": "integer"}, "description": "The numbers of the candidates whose words state another value for this property, or say nothing that states it."},
            "unclear_in": {"type": "array", "items": {"type": "integer"}, "description": "The numbers of the candidates whose words leave it open (the value listed among others, two places on the page disagreeing)."}},
        "required": ["requirement", "stated_by", "missed_by", "unclear_in"]}},
}, ["reason", "requirements"])

DECIDE_INSTRUCTIONS = """You are the final checker of a shopping assistant. A shopper wrote a request for one item and, while the assistant shopped, answered its questions and sent further lines. You are given the shopper's own words in order, the assistant's notes on them, and the store listings the assistant could put in the cart now. You list every requirement on the item that the shopper's words set, and under each requirement you sort every candidate: its words state the value, they miss it, or they leave it unclear. The assistant chooses from your lists. The shopper trusts the assistant only if the item meets every requirement their words set; one requirement read wrongly, or dropped, loses that trust. Read exactly what the words say; never assume, guess or fill in what they do not say.

WHAT YOU ARE GIVEN - each part in its own tag
- <task>: first cart (the first item to go into the cart) or order (the item to order).
- <shopper_words>: the only authority on what is wanted.
  - <request>: the shopper's request as first written. It is where every requirement starts.
  - <conversation>: every <turn> in which the shopper spoke, in the order they came. In a turn, <assistant> is what the assistant wrote (a question, a reply) and <shopper> is the shopper's line. A line answers the question in its own turn, or tells something new. A later line changes the request only as far as its own words say, and when two lines disagree the later one holds.
- <assistant_notes>: the assistant's record of those words - the <item>, each <requirement>, the <spending_limit> in force, the <priority> between fitting items and a wish <set_aside>. Check every note against <shopper_words>: where they differ, the shopper's words win.
- <candidates>: numbered <candidate> listings. Each gives its <title>, <brand> and <shelf>, the <variant_to_order> (its <options>, its <price> and its <stock> at the last reading), the listing's <facts> as the store files them, and its <other_variants>. Every candidate is in stock and within the spending limit at the last reading.

WHAT YOU RETURN
- reason: written first, in two to four sentences: the requirements the shopper's words set as they stand now, which notes differ from those words, and how the candidates compare.
- requirements: one entry per requirement on the item: the requirement as '<property>: <value>', then every candidate's number in exactly one of stated_by, missed_by and unclear_in.

RULES
- Judge each candidate only from its own words shown here - its options, its facts and its title. Never decide a value from the item's name or brand, or from what such a thing usually is or has.
- Check every candidate against every requirement: put each candidate's number under exactly one of stated_by, missed_by or unclear_in for each requirement, and leave no candidate out.
- A value a candidate's words do not state is missed, not assumed; when the words leave it open, use unclear_in, not stated_by.
- A candidate is fit only when it is under stated_by for every requirement. When none is, the assistant orders nothing - that is handled for you, so sort strictly rather than force a near one in.

STEP 1 - THE REQUIREMENTS IN FORCE. Read the shopper's words from first to last and list every requirement on the item itself that holds at the end.
 a. The request. Every detail it gives about the item: what it is made of, its colour, an amount with its unit (how much it holds; how big, long, heavy or strong it is; how many pieces come together), its pattern, shape, style, finish, kind or type, and what it must fit, match, include or contain.
 b. Answers. Every value the shopper gave in answer to a question is a requirement. An answer takes nothing back unless it says so.
 b2. A value is never the property's own name or a word of it, nor the item's own name said again: when a line names a thing together with the property or kind it belongs to, the thing named is the value of that property.
 c. Two values for one property. When the shopper requires a second value for a property that already has one (a second style, a second concern, a second feature), both hold: list both, each as its own entry.
 d. Replacements. A line that puts a new value in place of an earlier one (instead, forget, change, switch, replace, no longer, rather than) replaces that one value and nothing else. The new value is a requirement; the old one is gone.
 e. Additions. A line that says the item must also have or be something adds it.
 f. Withdrawals. A value taken back with nothing in its place is gone.
 g. Two doubted wishes. When the request doubts one item can offer two wishes, the one the shopper said matters more is a requirement and the other is not; when the shopper said both, both are; until then neither is.
 h. The first-cart task. You are given the request alone: list the requirements it states about the item, leaving out two wishes it doubts one item can offer together.
 Never list as a requirement: the kind of item itself (the store's shelf decides it); the price or the spending limit; being in stock, available or ready; the store's category or shelf; the shopper's priority between fitting items (the most, the least, the largest or the cheapest of something); why or for whom the shopper buys; a wish set aside. The assistant handles all of these itself, and an entry for one of them is wrong.

STEP 2 - SORT EVERY CANDIDATE UNDER EVERY REQUIREMENT, for the exact variant that would be ordered.
 Where a value counts, the surest first:
  1. The variant's options. They describe the exact item sold: a colour, a size, an amount, a character, a design, a scent, a flavour or a type they name decides that property, whatever the title or a fact lists (a title often lists the colours or sizes of every variant, and a fact is often filled in for the whole listing).
  2. A fact whose label names the property, when it states one value.
  3. The title, when it names the value for the item itself.
 stated_by - the candidate's words state the value:
  - the same word, another accepted spelling of it, a plain synonym, or a narrower kind of it (a specific kind of the wanted material, type or style is that material, type or style); a word with letters missing or out of place is unclear, not stated;
  - an amount: the same amount in any unit or notation (convert it yourself; sellers round), or a range written for this variant that holds it; never a deal (buy <n> take <m>, or its short form) read as a count of pieces, never '<n> in 1' read as <n> pieces, and never a figure of another measure of the item (an opening, a cut-out, a hole, a thread) or of a part sold with it;
  - a colour: that colour, or a light or dark shade of it; the colour's own word in the options beside words that name no colour (a thing, a flavour, a scent, a model) still states it, and so does the colour's word joined to what the item itself gives out or leaves behind when used (its glow, its mark): that colour is the variant's;
  - a material: the material's word put before the name of the item itself, in the title or the options (a <material> <item>), states that the item is made of it; before the name of another thing (a case for <material> things) it does not.
 missed_by - the candidate's words do not state it:
  - the options name another value of the property (another colour, size, amount, character, design, scent or type); the property's own fact states another value;
  - a colour joined in one name to another colour or to a metal is that other colour, not the wanted one; a colour word put before the wanted colour (<colour> <wanted colour>) names a colour of its own, unlike light, dark, pale or deep, which only name a shade of it;
  - a figure that belongs to another variant: the title lists several, or another variant's options name the wanted one while this variant's options name something else;
  - one figure of a measurement of several dimensions (length x width x height) states only the dimension it stands for;
  - a value said only of something else: an accessory, a free gift, the packaging, another item of a set, or what the item is compatible with (a value said of a part of the item itself - its handle, its strap, its base - is said of the item);
  - a value said only by a fact for another property: a fact states a value only when its own label names the property asked about;
  - nothing on the page states it.
 unclear_in - the candidate's words leave it open:
  - the property's own fact lists several values with the wanted one among them, and nothing else on the page says which one this item is;
  - the title names the wanted value among other values of the same property, and the options do not say which;
  - the title gives one amount while naming several variants, or several items sold under one name, and the options do not say that this variant is the one that amount belongs to;
  - two places on the page disagree: the options say one value and the property's own fact or the title another;
  - the title calls the item itself by another value of the property than its fact or its options give (a fact that says one material while the title names the item as made of another).

STEP 3 - CHECK YOUR LISTS. Under every requirement, every candidate number appears exactly once. A candidate is chosen only when it is under stated_by for every requirement, so sort strictly: when in doubt, unclear_in."""

SEAT_DECIDE = ModelSeat('decide', "Every requirement on the item, with the candidates that state it.", DECIDE_INSTRUCTIONS, DECIDE_SCHEMA, 4096, (GLM, GLM, LUNA))
SEAT_DECIDE_THINKING = ModelSeat('decide', "Every requirement on the item, with the candidates that state it.", DECIDE_INSTRUCTIONS, DECIDE_SCHEMA, 8192,
                                 (GLM_DEEP, GLM_FLASH, GLM))

STORE_SCHEMA = Schema.fields({
    "reason": {"type": "string", "description": "Written first, in one or two sentences: what the wanted value means, and which kinds of store entries state it."},
    "marks": Schema.strings("One entry per store entry, in the order shown: '<number> <same|kind|other|unclear>'."),
}, ["reason", "marks"])

STORE_INSTRUCTIONS = """You read a store's own words for a shopping assistant. A shopper requires one value for one property of an item. Stores write a value in their own words - another spelling, a synonym, a technical word, an abbreviation, a specific kind of it, the same amount in another unit - and the assistant can find and check listings only by the words the store actually uses. You are given the entries this store's listings use for that property (fact entries and the names of variants) and you say which of them state the wanted value.

WHAT YOU ARE GIVEN
The item, the property, the wanted value, the shopper's words that set it, and numbered store entries.

WHAT YOU RETURN
- reason: written first, in one or two sentences: what the wanted value means, and which kinds of entries state it.
- marks: one entry per store entry, in the order shown, '<number> <mark>':
  same - the entry is the wanted value itself, written another way: another spelling of the same words (another country's spelling; joined, hyphenated or spaced), an abbreviation sellers use for it, the same amount in another unit or notation, or a word that means exactly the same thing;
  kind - the entry names a narrower kind that is always the wanted value: every item of that kind is of the wanted value (a specific material that belongs to the general material asked for);
  other - the entry states a different value of the property, a wider kind of which the wanted value is only one, a member or an example of a group the wanted value names (one of its items, people or models), a neighbouring value, or something that only goes with the wanted value;
  unclear - the entry names the wanted value together with other values of the property, or cannot be told without more of the page.

HOW TO READ AN ENTRY - by its meaning, not its letters:
1. Two entries that share a word can mean different values, and an entry that shares no word with the wanted value can mean the same value.
2. Colours: a light or dark shade of the wanted colour is same; the colour joined in one name to another colour or to a metal is other; a pattern of several colours is unclear; a seller's code beside the colour word does not change the colour.
3. Materials: a specific material that belongs to the general material the shopper named is kind; a blend of the wanted material with another is unclear; a coating, a finish or a part made of it is other.
4. Amounts: convert the units yourself, exactly. The same quantity written in another unit is same only when the conversion is exact; a figure that only comes near it (rounded to a neighbouring figure) is other; a range that holds the amount is unclear; a different amount, or an amount of a different measure, is other. A deal (buy <n> take <m>, its short form, <n> plus <m> free) is a deal, not an amount: other. A count of something other than what the shopper counts (<n> colours, <n> models, <n> uses in one) is other.
5. A use, a concern, a benefit or a feature: an entry that names the same thing in other words is same - a technical word for it, or the problem named where the shopper named the remedy, or the reverse.
6. A type, a style, a theme or a name: another spelling or another name for exactly the same type is same; a narrower kind is kind only when every item of it is of the wanted type; a member of a group, a style of one place or period, and a neighbouring type are other.
7. The test for same and kind: would every listing filed under the entry be of the wanted value? When not every one would be, mark other or unclear.
8. Never mark same or kind only because the entry often goes with the wanted value.
Mark every entry, in order."""

SEAT_STORE = ModelSeat('store_words', "Which of the store's entries state a required value.", STORE_INSTRUCTIONS, STORE_SCHEMA, 8192, (GLM_FLASH, GLM_FLASH, GLM))


class Judge(Part):
    def __init__(self) -> None:
        self.verdicts: Dict[Key, Dict[str, Any]] = {}
        self.judged_under = ""
        self.judged_count = 0
        self.judge_room = JUDGE_LIMIT
    def settle(self) -> None:
        if self.clock.overdue() and self.verdicts:
            return
        shelf_words = set(Text.label_key(self.catalog.named_shelf()).split())
        kinds = [set(Text.label_key(part).split()) for part in re.split(r"\s*(?:&|,|/|\band\b)\s*", Text.lowered(self.catalog.named_shelf())) if Text.label_key(part).strip()]
        def the_shelf_itself(r: Dict[str, Any]) -> bool:
            words = set(Text.label_key(r["value"]).split())
            if len(kinds) >= 2 and any(words <= kind for kind in kinds) and not all(kind <= words for kind in kinds):
                return False
            return bool(r.get("stated") and shelf_words and words <= shelf_words)
        self.order.requirements = [r for r in self.order.requirements if not the_shelf_itself(r)]
        self.order.opening = [r for r in self.order.opening if not the_shelf_itself(r)]
        self.name_check()
        stamp = json.dumps([self.order.requirements, self.order.priority, self.matcher.lenient, len(self.matcher.named_as)], sort_keys=True)
        if stamp != self.judged_under:
            self.verdicts, self.judged_under = {}, stamp
        for key in self.sieve.candidates():
            if key not in self.verdicts:
                in_shelf = self.catalog.in_category(self.catalog.pages[key[0]])
                met = [self.matcher.settled(key, r) or (self.matcher.lenient and self.matcher.plain(key, r) >= 1) for r in self.order.requirements]
                self.verdicts[key] = {"kind_ok": in_shelf, "met": met, "by": "words"}
    def ask_reader(self) -> None:
        open_keys = [k for k in self.verdicts if self.sieve.affordable(k) and self.verdicts[k].get("by") == "words" and self.verdicts[k].get("kind_ok")
                     and not self.sieve.fits_words(k) and not any(self.matcher.plain(k, r) == -1 for r in self.order.requirements)]
        open_keys.sort(key=lambda k: -sum(self.matcher.plain(k, r) for r in self.order.requirements))
        batch = open_keys[:JUDGE_CARDS]
        if not batch or not self.desk.can_think() or self.judged_count >= self.judge_room:
            return
        self.judged_count += len(batch)
        wanted = "\n".join(f"{i}. {r['property']}: {r['value']}" for i, r in enumerate(self.order.requirements, start=1)) or "(none beyond the kind of thing)"
        text = (f"The shopper's own words, in order (what each requirement means):\n{self.episode.shopper_words()}\n\n"
                f"The shopper wants: {self.order.item}\nRequirements, in order:\n{wanted}\n"
                + (f"Priority between fitting items: {self.order.priority}\n" if self.order.priority else "")
                + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(batch, start=1)))
        answer = self.desk.ask(SEAT_JUDGE, text) or {}
        marks = answer.get("marks")
        marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
        for mark in marks:
            found = re.match(r"\s*\[?(\d+)\]?\W+([KkXx])\W*([YyNn\s,]*)", str(mark))
            if not found or not 1 <= int(found.group(1)) <= len(batch):
                continue
            key = batch[int(found.group(1)) - 1]
            said = [x in "Yy" for x in re.findall(r"[YyNn]", found.group(3))]
            said += [False] * (len(self.order.requirements) - len(said))
            met = [self.matcher.settled(key, r) or (said[i] and self.matcher.plain(key, r) != -1 and self.matcher.kind(r) != "colour"
                                                    and (not re.search(r"\d", r["value"]) or self.sieve.figure_on_page(key, r)))
                   for i, r in enumerate(self.order.requirements)]
            self.verdicts[key] = {"kind_ok": found.group(2) in "Kk", "met": met, "by": "reader"}
        for key in batch:
            if self.verdicts[key].get("by") == "words":
                self.verdicts[key]["by"] = "reader"
        for proof in answer.get("proof") or []:
            if isinstance(proof, dict) and isinstance(proof.get("n"), int) and 1 <= proof["n"] <= len(batch):
                self.verdicts[batch[proof["n"] - 1]].update({k: proof.get(k) for k in ("where", "words", "rank_value")})
        self.planner.add_queries(answer.get("queries") or [])
    def card(self, number: int, key: Key) -> str:
        page = self.catalog.pages[key[0]]
        named_words = {w for r in self.order.requirements for w in Text.name_words(r["property"])} | Text.name_words(self.order.priority)
        value_words = {w for r in self.order.requirements for w in Text.stems(r["value"])}
        close, other = [], []
        for label, value in (page.get("facts") or {}).items():
            if Text.name_words(label) & named_words or Text.stems(value) & value_words:
                close.append(f"{label} = {str(value)[:110]}")
            else:
                other.append(f"{label} = {str(value)[:36]}")
        facts = "; ".join(close + other[:5])
        for position, requirement in enumerate(self.order.requirements, start=1):
            named = self.matcher.named_facts(page, requirement)
            if named:
                facts += f" || the fact named for requirement {position}: " + "; ".join(f"{label} = {value[:110]}" for label, value in named)
        price = (self.catalog.stock.get(key) or {}).get("price")
        return (f"[{number}] title: {str(page.get('title') or '')[:150]} | options: {self.catalog.variant_of(key).get('options') or '-'} | price: {price} | "
                f"category: {' > '.join(str(x) for x in (page.get('category_path') or [])[-2:])} | brand: {page.get('brand') or '-'} | facts: {facts or '-'}")
    def name_check(self) -> None:
        for requirement in self.order.requirements:
            kind = self.matcher.kind(requirement)
            if kind not in ("colour", "material") or not self.desk.can_think() or self.matcher.name_checks >= 10:
                continue
            others = [r for r in self.order.requirements if r is not requirement]
            waiting = []
            for key in self.sieve.candidates():
                tag = (key, requirement["property"], requirement["value"])
                if tag in self.matcher.named_as or any(self.matcher.plain(key, r) < 1 for r in others):
                    continue
                page = self.catalog.pages.get(key[0]) or {}
                options = str(self.catalog.variant_of(key).get("options") or "").strip()
                if kind == "colour":
                    texts = [f"variant: {options}" if options else f"title: {str(page.get('title') or '')[:150]}"]
                else:
                    texts = [f"title: {str(page.get('title') or '')[:150]}"] + ([f"variant: {options}"] if options else [])
                waiting.extend((tag, text) for text in texts)
            waiting = waiting[:16]
            if not waiting:
                continue
            self.matcher.name_checks += 1
            asked = (f"Item: {self.order.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n\nTexts:\n"
                     + "\n".join(f"[{i}] {text}" for i, (_, text) in enumerate(waiting, start=1)))
            answer = self.desk.ask(SEAT_NAMES, asked) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            numbered = all(re.match(r"\s*\[?\d+", str(mark)) for mark in marks)
            if numbered or len(marks) == len(waiting):
                for position, mark in enumerate(marks, start=1):
                    word = re.search(r"\b(only|among|other|none)\b", Text.lowered(mark))
                    number = re.match(r"\s*\[?(\d+)", str(mark))
                    at = int(number.group(1)) if numbered else position
                    if word and 1 <= at <= len(waiting):
                        tag, text = waiting[at - 1]
                        if text.startswith("variant: ") and word.group(1) != "none" or tag not in self.matcher.named_as:
                            self.matcher.named_as[tag] = word.group(1)
            for tag, _ in waiting:
                self.matcher.named_as.setdefault(tag, "unread")
            self.matcher.read_cache.clear()
    @staticmethod
    def in_order(entries: List[str], count: int) -> List[str]:
        numbered = [re.match(r"\s*\[?(\d+)\]?[.:)]?\s+(?:(-?\d+(?:\.\d+)?)(?![\d.])|(none)\b)", entry, re.I) for entry in entries]
        if len(entries) != count or not entries or (all(numbered) and sorted(int(m.group(1)) for m in numbered if m) == list(range(1, count + 1))):
            return entries
        out: List[str] = []
        for place, (entry, found) in enumerate(zip(entries, numbered), start=1):
            if found and int(found.group(1)) == place:
                out.append(entry)
                continue
            lead = re.match(r"\s*(-?\d+(?:\.\d+)?)(?![\d.])", entry)
            if re.match(r"\s*(?:-?\d+(?:\.\d+)?\s+)?none\b", entry, re.I) or not lead:
                out.append(f"{place} none")
                continue
            quoted = re.search(r'"(.{1,80})"', entry[lead.end():])
            words = quoted.group(1) if quoted else " ".join(entry.replace('"', " ").split())
            out.append(f'{place} {lead.group(1)} "{words}"')
        return out
    def rank_by_priority(self, every: bool = False) -> None:
        if not self.order.priority or self.sieve.priced_priority():
            return
        for _ in range(4 if every else 1):
            fitting = self.sieve.ranked()
            keys = [k for k in fitting if k not in self.sieve.figures and not (k[0] in self.catalog.pages and self.sieve.counted_figure(k)[0])][:JUDGE_CARDS]
            if not keys or not self.desk.can_think() or len(self.sieve.figures) >= self.sieve.figure_room or (len(fitting) < 2 and not every):
                return
            text = (f"Item: {self.order.item}\nThe shopper's priority: {self.order.priority}\n"
                    + (f"Write every figure in this unit, as for the listings read before: {self.sieve.figure_unit}\n" if self.sieve.figure_unit else "")
                    + "\nListings:\n" + "\n".join(self.card(i, key) for i, key in enumerate(keys, start=1)))
            answer = self.desk.ask(SEAT_RANK, text) or {}
            if str(answer.get("better") or "").lower() in ("higher", "lower"):
                self.sieve.better = str(answer["better"]).lower()
            if not self.sieve.figure_unit and isinstance(answer.get("unit"), str):
                self.sieve.figure_unit = " ".join(answer["unit"].split())[:20]
            values = answer.get("values")
            values = re.split(r"[\n;]+", values) if isinstance(values, str) else (values or [])
            entries = self.in_order([str(entry) for entry in values], len(keys))
            numbered = [re.match(r"\s*\[?(\d+)\]?\W+(-?\d+(?:\.\d+)?)", entry) for entry in entries]
            if not any(numbered) and len(entries) == len(keys):
                for key, entry in zip(keys, entries):
                    bare = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*", entry)
                    if bare:
                        self.sieve.figures[key] = float(bare.group(1))
            else:
                for found, entry in zip(numbered, entries):
                    quoted = re.search(r'"(.{1,80})"\s*$', entry) or re.search(r'"(.{1,80})"', entry)
                    alone = re.match(r"\s*(-?\d+(?:\.\d+)?)", entry) if len(keys) == len(entries) == 1 and quoted else None
                    if not (found and 1 <= int(found.group(1)) <= len(keys)) and alone and Text.says_amount(quoted.group(1), float(alone.group(1))):
                        found = re.match(r"(\d+) (-?\d+(?:\.\d+)?)", "1 " + alone.group(1))
                    if found and 1 <= int(found.group(1)) <= len(keys):
                        key = keys[int(found.group(1)) - 1]
                        if quoted and self.sieve.figure_unit and not Text.shows_amount(quoted.group(1), self.sieve.figure_unit):
                            page = self.catalog.pages.get(key[0]) or {}
                            texts = [page.get("title"), self.catalog.variant_of(key).get("options")] + [f"{label} {value}" for label, value in (page.get("facts") or {}).items()]
                            if not Text.unit_elsewhere(quoted.group(1), texts, self.sieve.figure_unit):
                                self.sieve.figures[key] = None
                                continue
                        self.sieve.figures[key] = float(found.group(2))
                        if quoted:
                            self.sieve.figure_words[key] = quoted.group(1)
            for key in keys:
                self.sieve.figures.setdefault(key, None)


class Episode(Part):
    def __init__(self, problem: Dict[str, Any]) -> None:
        clock = Clock()
        session = StoreSession(problem)
        catalog = Catalog()
        desk = ModelDesk(session.spread)
        order = Order(session.query)
        reader, planner, interviewer = Reader(), SearchPlanner(), Interviewer()
        matcher, sieve, judge = Matcher(), Sieve(), Judge()
        self.early_tries = 0
        self.early_seen = 0
        self.first_marks: Dict[Key, str] = {}
        self.due_turn = 0
        self.stand_in: Optional[Key] = None
        self.named_nearest: Optional[Key] = None
        self.looked_under = ""
        self.decide_asks = 0
        self.decide_hunted = False
        self.late_opens = 0
        self.near_opens = 0
        self.record_repairs = 0
        self.store_asks = 0
        self.store_asked: Dict[Tuple[str, str], Set[str]] = {}
        self.swept: Dict[str, int] = {}
        for part in (session, catalog, desk, order, reader, planner, interviewer, matcher, sieve, judge, self):
            part.wire(clock, session, catalog, desk, order, reader, planner, interviewer, matcher, sieve, judge, self)
    def shopper_lines(self) -> List[Tuple[str, str]]:
        out = []
        for entry in self.session.dialogue:
            wrote = " ".join(str((a.get("args") or {}).get("content") or "").strip() for a in entry.get("tool_calls") or [] if a.get("name") == "message").strip()
            result = entry.get("environment_result")
            said = result.get("user_message") if isinstance(result, dict) else None
            line = str(said.get("content") or "").strip() if isinstance(said, dict) else ""
            if line:
                out.append((wrote, line))
        return out
    def shopper_words(self) -> str:
        out = [f'Request: "{self.order.request}"']
        try:
            for number, (wrote, line) in enumerate(self.shopper_lines(), start=1):
                out.append(f'{number}. ' + (f'The assistant wrote: "{wrote[:400]}" - the shopper: "{line[:600]}"' if wrote else f'The shopper: "{line[:600]}"'))
        except Exception:
            pass
        return "\n".join(out)
    @staticmethod
    def tag_safe(text: Any, limit: int) -> str:
        return " ".join(str(text or "").split())[:limit].replace("<", "(").replace(">", ")")
    def tagged_words(self, first: bool) -> str:
        out = [f"<request>{self.tag_safe(self.order.request, 2000)}</request>"]
        if first:
            return "\n".join(out)
        turns = []
        try:
            for number, (wrote, line) in enumerate(self.shopper_lines(), start=1):
                said = f"<assistant>{self.tag_safe(wrote, 400)}</assistant>" if wrote else ""
                turns.append(f'<turn number="{number}">{said}<shopper>{self.tag_safe(line, 600)}</shopper></turn>')
        except Exception:
            turns = []
        out.append("<conversation>" + ("\n" + "\n".join(turns) + "\n" if turns else "") + "</conversation>")
        return "\n".join(out)
    def listing_text(self, number: int, key: Key) -> str:
        page = self.catalog.pages.get(key[0]) or {}
        seen = self.catalog.stock.get(key) or {}
        variant = self.catalog.variant_of(key)
        facts = "; ".join(f"{label} = {str(value)[:160]}" for label, value in (page.get("facts") or {}).items())
        others = []
        for other in page.get("variants") or []:
            if str(other.get("sku")) == key[1]:
                continue
            read = self.catalog.stock.get((key[0], str(other.get("sku")))) or {}
            stock = read.get("in_stock", other.get("in_stock"))
            others.append(f'"{other.get("options")}" {read.get("price", other.get("price"))}' + (" sold out" if stock is False else ""))
        shelf = " > ".join(str(x) for x in (page.get("category_path") or [])[-2:])
        safe = self.tag_safe
        return (f'<candidate number="{number}">\n'
                f"<title>{safe(page.get('title'), 200)}</title> <brand>{safe(page.get('brand') or '-', 80)}</brand> <shelf>{safe(shelf or '-', 120)}</shelf>\n"
                f"<variant_to_order><options>{safe(variant.get('options') or '-', 160)}</options> <price>{seen.get('price', variant.get('price'))}</price> "
                f"<stock>{'sold out' if seen.get('in_stock') is False else 'in stock'}</stock></variant_to_order>\n"
                f"<facts>{safe(facts or '-', 1500)}</facts>" + (f"\n<other_variants>{safe('; '.join(others[:10]), 800)}</other_variants>" if others else "")
                + "\n</candidate>")
    def decide(self, candidates: List[Key], first: bool = False) -> Tuple[Optional[Key], bool]:
        try:
            return self._decide(candidates, first)
        except Exception:
            return None, False
    def _decide(self, candidates: List[Key], first: bool) -> Tuple[Optional[Key], bool]:
        keys = [k for k in dict.fromkeys(candidates) if k in self.judge.verdicts and self.sieve.on_sale(k)][:DECIDE_CARDS]
        if not keys or self.decide_asks >= DECIDE_LIMIT + (0 if first else DECIDE_SPARE) or self.clock.now() > DECIDE_STOP or not self.desk.can_think():
            return None, False
        self.decide_asks += 1
        held = (self.order.opening or self.order.requirements) if first else self.order.requirements
        safe = self.tag_safe
        notes = [f"<item>{safe(self.order.item, 120)}</item>"]
        notes += [f"<requirement>{safe(r['property'], 60)} = {safe(r['value'], 80)}</requirement>" for r in held] or ["<requirement>(none)</requirement>"]
        notes.append(f"<spending_limit>{self.order.budget if self.order.budget is not None else 'none'}</spending_limit>")
        if self.order.priority and not first:
            notes.append(f"<priority>{safe(self.order.priority, 160)}</priority>")
        aside = [] if first else [r for r in self.order.set_aside if not any(Reader.same_value(r["value"], x["value"]) for x in self.order.requirements)]
        if aside:
            notes += [f"<set_aside>{safe(r['property'], 60)} = {safe(r['value'], 80)}</set_aside>" for r in aside]
        text = (f"<task>{'first cart' if first else 'order'}</task>\n\n<shopper_words>\n{self.tagged_words(first)}\n</shopper_words>\n\n"
                "<assistant_notes>\n" + "\n".join(notes) + "\n</assistant_notes>\n\n"
                "<candidates>\n" + "\n".join(self.listing_text(i, k) for i, k in enumerate(keys, start=1)) + "\n</candidates>")
        answer = self.desk.ask(SEAT_DECIDE_THINKING, text)
        if not answer or not self.checked_requirements(answer):
            answer = self.desk.ask_agreed(SEAT_DECIDE, text, self.decide_key, merge=self.decide_merge) or {}
        listed = self.checked_requirements(answer)
        marks: Dict[int, str] = {}
        for number in range(1, len(keys) + 1):
            if any(number in missed for _, _, missed in listed):
                marks[number] = "N"
            elif listed and all(number in stated for _, stated, _ in listed):
                marks[number] = "Y"
            else:
                marks[number] = "?"
        result: Tuple[Optional[Key], bool] = (None, False)
        if listed and first:
            self.first_marks.update({key: marks[number] for number, key in enumerate(keys, start=1)})
        if listed:
            if first:
                after = next((n for n in range(1, len(keys) + 1) if marks[n] == "Y"), None)
                result = (keys[after - 1], False) if after else (None, True)
            elif (marks[1] == "Y" or (marks[1] == "?" and not any(m == "Y" for m in marks.values())) or self.firm_misses(keys[0], listed)
                  or (marks[1] == "?" and self.options_settle(keys[0], listed))):
                result = (keys[0], False)
            elif marks[1] == "?":
                fitting = set(self.sieve.ranked())
                least = self.sieve.trust(keys[0]) - 1e-9
                after = next((n for n in range(2, len(keys) + 1) if marks[n] == "Y" and keys[n - 1] in fitting and self.sieve.trust(keys[n - 1]) >= least), None)
                result = (keys[after - 1] if after else keys[0], False)
            else:
                fitting = [k for k in keys if k in set(self.sieve.ranked())]
                after = next((n for n in range(2, len(keys) + 1) if marks[n] == "Y" and keys[n - 1] in fitting), None)
                if after is None and not fitting:
                    after = next((n for n in range(2, len(keys) + 1) if marks[n] == "Y"), None)
                result = (keys[after - 1], False) if after else (None, True)
        if not first and listed:
            self.repair_record([t for t, _, _ in listed])
        return result
    def firm_misses(self, key: Key, listed: List[Tuple[str, Set[int], Set[int]]]) -> bool:
        missed = [text for text, _, miss in listed if 1 in miss]
        if not missed:
            return False
        for text in missed:
            words = set(Text.word_list(Text.lowered(text)))
            match = [r for r in self.order.requirements if set(Text.word_list(Text.lowered(r["value"]))) <= words]
            if not match or any(self.sieve.reliance(key, r) < FIRM_FIT for r in match):
                return False
        return True
    def options_settle(self, key: Key, listed: List[Tuple[str, Set[int], Set[int]]]) -> bool:
        options = str(self.catalog.variant_of(key).get("options") or "").strip()
        unclear = [text for text, stated, missed in listed if 1 not in stated and 1 not in missed]
        if not options or not unclear:
            return False
        for text in unclear:
            words = set(Text.word_list(Text.lowered(text)))
            match = [r for r in self.order.requirements if set(Text.word_list(Text.lowered(r["value"]))) <= words]
            if not match or not all(self.matcher.says(options, r) and not self.matcher.options_name_other(key, r) for r in match):
                return False
        return True
    @staticmethod
    def decide_merge(readings: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        fields = ("stated_by", "missed_by", "unclear_in")
        def value_of(entry: Dict[str, Any]) -> str:
            text = " ".join(str(entry.get("requirement") or "").split())
            return " ".join(Text.word_list(text.split(":", 1)[-1] if ":" in text else text))
        def numbers(entry: Dict[str, Any], field: str) -> Set[int]:
            got: Set[int] = set()
            for item in entry.get(field) or []:
                try:
                    got.add(int(item))
                except (TypeError, ValueError):
                    continue
            return got
        tables = []
        for reading in readings:
            table: Dict[str, Dict[str, Any]] = {}
            for entry in reading.get("requirements") or []:
                if isinstance(entry, dict) and value_of(entry):
                    table.setdefault(value_of(entry), entry)
            tables.append(table)
        merged = []
        for value in dict.fromkeys(v for table in tables for v in table):
            holders = [table[value] for table in tables if value in table]
            if len(holders) * 2 <= len(tables):
                continue
            entry: Dict[str, Any] = {"requirement": holders[0].get("requirement"), "stated_by": [], "missed_by": [], "unclear_in": []}
            for number in sorted(set().union(*(numbers(h, f) for h in holders for f in fields))):
                votes = [f for h in holders for f in fields if number in numbers(h, f)]
                best = max(fields, key=votes.count)
                entry[best if votes.count(best) * 2 > len(holders) else "unclear_in"].append(number)
            merged.append(entry)
        return {"requirements": merged} if merged else None
    def decide_key(self, answer: Dict[str, Any]) -> Any:
        return frozenset((Text.lowered(text), frozenset(stated), frozenset(missed)) for text, stated, missed in self.checked_requirements(answer or {}))
    def checked_requirements(self, answer: Dict[str, Any]) -> List[Tuple[str, Set[int], Set[int]]]:
        out = []
        for entry in answer.get("requirements") or []:
            if not isinstance(entry, dict):
                continue
            text = " ".join(str(entry.get("requirement") or "").split())
            found = re.match(r"\s*([^:=]{2,60}?)\s*[:=]\s*(.{1,80}?)\s*$", text)
            if not found or not self.of_the_item(found.group(1), found.group(2)):
                continue
            numbers = []
            for name in ("stated_by", "missed_by"):
                got: Set[int] = set()
                for item in entry.get(name) or []:
                    try:
                        got.add(int(item))
                    except (TypeError, ValueError):
                        continue
                numbers.append(got)
            out.append((text, numbers[0], numbers[1]))
        return out
    def of_the_item(self, prop: str, value: str) -> bool:
        text = Text.lowered(f"{prop} {value}")
        if NOT_A_PROPERTY.search(text) or re.search(r"\b(?:spend\w*|limit|costs?|afford\w*|priorit\w*)\b", text):
            return False
        ranked = {w for w in Text.content_words(self.order.priority) if not re.search(r"\d", w)}
        if ranked and (Text.content_words(value) - Text.name_words(prop)) & ranked:
            return False
        if re.search(r"\b(?:purpose|reason|recipient|gift)\b", Text.lowered(prop)):
            return False
        own = Text.content_words(f"{self.order.item} {self.order.category} {self.catalog.named_shelf()}")
        words = Text.content_words(value)
        kinds = [Text.content_words(part) for name in (self.order.item, self.catalog.named_shelf())
                 for part in re.split(r"\s*(?:&|,|/|\band\b)\s*", Text.lowered(name)) if Text.content_words(part)]
        picks = len(kinds) >= 2 and words and any(words <= kind for kind in kinds) and not all(kind <= words for kind in kinds)
        if words and own and words <= own and not picks:
            return False
        for purpose in self.order.purposes:
            own = Text.content_words(purpose)
            if words and own and 2 * len(words & own) >= len(words):
                return False
        return True
    def repair_record(self, listed: List[Any]) -> bool:
        lines = [line for _, line in self.shopper_lines()]
        said = " ".join([self.order.request] + lines)
        said_words = Text.stems(said) | set(Text.word_list(Text.lowered(said)))
        held = self.order.requirements + self.order.set_aside
        taken = {Text.closed_up(v) for v in self.order.took_back.values()} | self.order.dropped
        unchosen = {w for wish in getattr(self.order, "pair_wishes", []) or [] for w in Text.content_words(wish)
                    if not any(Text.content_words(r["value"]) & Text.content_words(wish) for r in self.order.requirements)}
        backs = [line for line in lines if TAKES_BACK.search(Text.lowered(line))]
        added = 0
        for entry in listed:
            found = re.match(r"\s*([^:=]{2,60}?)\s*[:=]\s*(.{1,80}?)\s*$", str(entry))
            if not found or added >= 2:
                continue
            prop, value = found.group(1).strip(), found.group(2).strip().strip("\"'")
            if not value or not self.of_the_item(prop, value) or Text.closed_up(value) in taken:
                continue
            if any(Reader.same_value(r["value"], value) or Text.closed_up(r["value"]) in Text.closed_up(value)
                   or Text.closed_up(value) in Text.closed_up(r["value"]) for r in held):
                continue
            words = [w for w in Text.word_list(Text.lowered(value)) if len(w) >= 2]
            if not words or not all(w in said_words or Text.stems(w) & said_words for w in words):
                continue
            if unchosen and Text.content_words(value) and Text.content_words(value) <= unchosen:
                continue
            if Text.says_value(self.order.request, value) and any(Text.says_value(line, value) for line in backs):
                continue
            self.order.requirements.append(self.order.requirement({"property": prop, "value": value}))
            added += 1
        return added > 0
    @staticmethod
    def spelled_alike(entry: str, value: str) -> bool:
        if re.sub(r"[\s\-]+", "", entry) == re.sub(r"[\s\-]+", "", value):
            return True
        mine, theirs = entry.split(), value.split()
        return len(mine) == len(theirs) and all(Text.same_word(a, b) or Text.one_letter_apart(a, b) for a, b in zip(mine, theirs))
    def store_entries(self, requirement: Dict[str, Any]) -> List[str]:
        related, inferred, _ = self.matcher.own_labels(requirement)
        kind = self.matcher.kind(requirement)
        figures = bool(re.search(r"\d", requirement["value"]))
        pages = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        counts: Dict[str, int] = {}
        for page in pages:
            for label, value in (page.get("facts") or {}).items():
                if str(label) in related or str(label) in inferred:
                    for entry in Text.entries(value):
                        counts[entry] = counts.get(entry, 0) + 1
            if kind in ("colour", "amount"):
                for variant in page.get("variants") or []:
                    for entry in Text.entries(variant.get("options")):
                        if len(entry.split()) <= 5:
                            counts[entry] = counts.get(entry, 0) + 1
        known = self.matcher.forms(requirement)
        own = {w for form in known for w in form.split()}
        entries = [e for e in counts if 2 <= len(e) <= 60 and bool(re.search(r"\d", e)) == figures and e not in known]
        def near(entry: str) -> bool:
            return any(Text.same_word(w, o) or Text.one_letter_apart(w, o) for w in entry.split() for o in own)
        return sorted(entries, key=lambda e: (not near(e), -counts[e], e))
    def read_store_words(self) -> bool:
        learnt = False
        for requirement in list(self.order.requirements):
            if self.store_asks >= STORE_LIMIT or not self.desk.can_think() or self.clock.late():
                break
            tag = (Text.lowered(requirement["property"]), Text.lowered(requirement["value"]))
            asked = self.store_asked.setdefault(tag, set())
            fresh = [e for e in self.store_entries(requirement) if e not in asked]
            if not fresh or (asked and len(fresh) < 3):
                continue
            batch = fresh[:STORE_ENTRIES]
            asked.update(batch)
            self.store_asks += 1
            said = next((line for _, line in reversed(self.shopper_lines()) if Text.says_value(line, requirement["value"])), self.order.request)
            text = (f"Item: {self.order.item}\nProperty: {requirement['property']}\nWanted value: {requirement['value']}\n"
                    f'The shopper\'s words that set it: "{said[:400]}"\n\nStore entries:\n' + "\n".join(f"[{i}] {e}" for i, e in enumerate(batch, start=1)))
            answer = self.desk.ask(SEAT_STORE, text) or {}
            marks = answer.get("marks")
            marks = re.split(r"[\n;,]+", marks) if isinstance(marks, str) else (marks or [])
            same: List[str] = []
            sought: List[str] = []
            other: List[str] = []
            material = self.matcher.kind(requirement) == "material" or "material" in Text.lowered(requirement["property"])
            material = material or self.matcher.picks_kind(requirement) is not None
            plain_value = " ".join(Text.singular(w) for w in Text.word_list(Text.closed_up(requirement["value"])))
            for mark in marks:
                found = re.match(r"\s*\[?(\d+)\]?\W+(same|kind|other|unclear)\b", Text.lowered(str(mark)))
                if not found or not 1 <= int(found.group(1)) <= len(batch):
                    continue
                entry = batch[int(found.group(1)) - 1]
                if found.group(2) in ("same", "kind"):
                    if re.search(r"\d", requirement["value"]):
                        amounts = re.findall(r"\d+(?:\.\d+)?\s*[a-z]+", entry)
                        given = [form for form in [requirement["value"]] + list(requirement.get("written") or []) + list(requirement.get("alike") or [])
                                 if re.search(r"\d\s*[a-z]", Text.lowered(form))]
                        if len(amounts) == 1 and any(Text.same_amount(form, amounts[0]) and Text.same_amount(amounts[0], form) for form in given):
                            same.append(amounts[0])
                    elif self.spelled_alike(entry, plain_value) or material and set(plain_value.split()) <= {Text.singular(w) for w in Text.word_list(entry)}:
                        same.append(entry)
                    else:
                        sought.append(entry)
                elif found.group(2) == "other" and not re.search(r"\d", requirement["value"]):
                    if set(entry.split()) & {w for form in self.matcher.forms(requirement) for w in form.split()}:
                        other.append(entry)
            for held in self.order.requirements + self.order.opening:
                if (Text.lowered(held["property"]), Text.lowered(held["value"])) != tag:
                    continue
                if same:
                    held["store"] = list(dict.fromkeys(list(held.get("store") or []) + same))[:12]
                if sought:
                    held["sought"] = list(dict.fromkeys(list(held.get("sought") or []) + sought))[:12]
                    held["alike"] = list(dict.fromkeys(list(held.get("alike") or []) + sought))[:16]
                if other:
                    held["refused"] = list(dict.fromkeys(list(held.get("refused") or []) + other))[:60]
            learnt = learnt or bool(same) or bool(other) or bool(sought)
        if learnt:
            self.matcher.read_cache.clear()
            self.judge.settle()
            self.planner.add_queries(self.planner.store_queries(), first=True)
        return learnt
    def sweep(self, force: bool = False) -> bool:
        stamp = json.dumps([[r["property"], r["value"]] for r in self.order.requirements] + [self.order.budget], default=str)
        done = self.swept.get(stamp, 0)
        if (done >= (3 if force else 2) or self.session.over or not self.session.has("search") or self.clock.late()
                or self.session.turns_left() <= TURN_RESERVE + 2):
            return False
        self.swept[stamp] = done + 1
        try:
            self.read_store_words()
        except Exception:
            pass
        spent = False
        for round_number in range(SWEEP_ROUNDS):
            if self.session.over or self.session.turns_left() <= TURN_RESERVE + 2 or self.clock.late():
                break
            fitting = [k for k in self.sieve.ranked() if k not in self.sieve.gone]
            if round_number and any(self.sieve.firm(k) for k in fitting[:6]):
                break
            self.planner.search_room = max(self.planner.search_room, len(self.planner.sent) + 2 * self.session.group_size)
            self.planner.open_room = max(self.planner.open_room, len(self.catalog.pages) + 3 * self.session.group_size)
            calls = self.planner.maker_filters(SWEEP_MAKERS)
            for query in self.planner.sweep_queries(round_number + 3 * done):
                text = Text.lowered(" ".join(query.split())[:QUERY_LENGTH])
                if len(calls) >= self.session.group_size:
                    break
                if text and text not in self.planner.sent:
                    self.planner.sent.add(text)
                    self.planner.searched.append(text)
                    calls.append(StoreSession.call("search", query=" ".join(text.replace('"', " ").split()), k=10, in_stock=True))
            if not calls:
                break
            self.session.send(calls)
            self.reader.hear_quietly()
            opens = self.planner.to_open(self.session.group_size * OPEN_BATCH, allowance=len(self.catalog.pages) + 3 * self.session.group_size)
            if opens and not self.session.over and self.session.turns_left() > TURN_RESERVE + 1:
                self.session.send(opens[: self.session.group_size])
                self.reader.hear_quietly()
            self.judge.settle()
            spent = True
            if round_number == 0:
                try:
                    self.read_store_words()
                except Exception:
                    pass
        if spent:
            self.sieve.ranked()
        return spent
    def best_held(self) -> Optional[Key]:
        live = [k for k in self.catalog.cart if k not in self.sieve.gone]
        held = next((k for k in self.sieve.last_ranked if k in live), None)
        if held is None and live and (self.clock.now() > STORE_STOP - 10 or self.session.turns_left() < 3):
            held = live[-1]
        return held
    def cart_early(self) -> bool:
        if (self.catalog.cart or self.early_tries >= 3 or not self.catalog.pages or self.session.over or self.clock.late()
                or not self.session.has("add_to_cart") or self.session.turns_left() < EARLY_TURNS + 4):
            return False
        if self.order.pair_fallback or self.order.pair_chosen or self.order.set_aside:
            return False
        if self.early_tries and len(self.catalog.pages) == self.early_seen:
            return False
        self.early_tries += 1
        self.early_seen = len(self.catalog.pages)
        self.judge.settle()
        key = self.early_choice()
        if key is None:
            return False
        group = [StoreSession.call("add_to_cart", product_id=key[0], sku=key[1])]
        if self.session.has("inspect_stock"):
            group.append(StoreSession.call("inspect_stock", product_id=key[0], sku=key[1]))
        group += self.planner.next_searches(self.session.group_size - len(group))
        self.session.send(group)
        if key not in self.catalog.cart:
            return True
        self.stand_in = key
        seen = self.catalog.stock.get(key) or {}
        price = seen.get("price")
        self.reader.hear()
        if seen.get("in_stock") is False or (self.order.budget is not None and isinstance(price, float) and price > self.order.budget + 1e-9):
            self.sieve.gone.add(key)
            self.lost_early(key)
            self.reader.pending_line = ""
        return True
    def early_choice(self) -> Optional[Key]:
        meeting = [k for k in self.judge.verdicts if self.sieve.affordable(k) and self.sieve.meets_opening(k)]
        firmly = [k for k in meeting if all(self.sieve.reliance(k, r) >= FIRM_FIT for r in self.order.opening)]
        keys = [k for k in firmly if self.sieve.doubts(k) == 0]
        if not keys and self.early_tries >= 2:
            keys = [k for k in meeting if self.sieve.doubts(k) == 0] or firmly
        if not keys and self.early_tries >= 3:
            keys = meeting
        if not keys:
            return None
        shelf = [p for p in self.catalog.pages.values() if self.catalog.in_category(p)] or list(self.catalog.pages.values())
        seen_in: Dict[str, int] = {}
        for page in shelf:
            for word in Text.content_words(str(page.get("title") or "")):
                seen_in[word] = seen_in.get(word, 0) + 1
        middle = Text.middle([(self.catalog.stock.get(k) or {}).get("price") or 0.0 for k in keys])
        def plainly(key: Key) -> Tuple[Any, ...]:
            price = (self.catalog.stock.get(key) or {}).get("price") or 0.0
            words = Text.content_words(str((self.catalog.pages.get(key[0]) or {}).get("title") or ""))
            usual = sum(seen_in.get(w, 0) for w in words) / max(1, len(words))
            return (-sum(self.matcher.plain(key, r) + self.matcher.exact(key, r) for r in self.order.opening), -round(usual, 1), abs(price - middle), price)
        def titled(key: Key) -> int:
            title = (self.catalog.pages.get(key[0]) or {}).get("title")
            return sum(1 for r in self.order.opening if self.matcher.says(title, r) or (self.matcher.kind(r) == "amount" and self.matcher.amounts(title, r)[0]))
        wanted_now = set(self.sieve.ranked())
        if not self.first_cart_due() and any(r not in self.order.opening for r in self.order.requirements) and not wanted_now & set(keys):
            return None
        firm = set(k for k in keys if all(self.sieve.reliance(k, r) >= FIRM_FIT for r in self.order.opening))
        ordered = sorted(keys, key=lambda k: (self.sieve.doubts(k), k not in firm, -titled(k), -self.sieve.own_exact(k), k not in wanted_now) + plainly(k))
        picked, none_fit = self.decide(ordered, first=True)
        if picked is not None:
            return picked
        if none_fit and not self.first_cart_due():
            return None
        return self.due_choice(ordered) if none_fit else ordered[0]
    def first_cart_due(self) -> bool:
        return (not self.catalog.cart and not self.session.over and self.session.has("add_to_cart")
                and self.session.turn >= max(6, (self.session.turn_limit * 2) // 5))
    def due_choice(self, keys: List[Key]) -> Optional[Key]:
        usable = [k for k in keys if self.first_marks.get(k) != "N"]
        rank = {"Y": 0, "?": 1}
        if usable:
            return min(usable, key=lambda k: (rank.get(self.first_marks.get(k, ""), 2), self.sieve.doubts(k), -self.sieve.own_exact(k)))
        return None
    def cart_due(self) -> bool:
        if self.catalog.cart or not self.first_cart_due() or self.due_turn == self.session.turn:
            return bool(self.catalog.cart)
        self.due_turn = self.session.turn
        try:
            self.cart_stand_in(force=True)
        except Exception:
            pass
        return bool(self.catalog.cart)
    def lost_early(self, key: Key) -> None:
        title = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title"))
        if not title or not self.sieve.fits_words(key):
            return
        self.sieve.lost_titles.add(title)
        words = Text.word_list(title)
        own = " ".join(w for w in Text.word_list(self.catalog.variant_of(key).get("options")) if not w.replace(".", "").isdigit())
        self.planner.add_queries([" ".join(words[:10])] + ([f"{own} {' '.join(words[:10])}"] if own else []), first=True)
    def settle_early(self) -> None:
        for _ in range(3):
            if self.catalog.cart or self.session.over or self.early_tries >= 3 or self.session.turns_left() < EARLY_TURNS + 4 or self.clock.late():
                break
            if not self.catalog.pages or self.early_tries:
                more = self.planner.to_open(self.session.group_size * OPEN_BATCH)[: self.session.group_size] or self.planner.next_searches(self.session.group_size)
                if not more:
                    break
                self.session.send(more)
                self.reader.hear_quietly()
            self.cart_early()
        self.reply_early()
    def reply_early(self) -> bool:
        if not self.order.unanswered or not self.catalog.cart or self.session.over or not self.session.has("message") or self.session.turns_left() <= EARLY_TURNS:
            return False
        lines = self.order.unanswered[-4:]
        said = "\n".join(f"- {line}" for line in lines)
        if not self.order.holds_now(said)[0] and not any(part.strip().endswith("?") for line in lines for part in re.split(r"(?<=[.!?])\s+", line)):
            self.order.unanswered = []
            return False
        room = self.session.group_size - 1
        work = self.planner.to_open(room * OPEN_BATCH)[:room] or self.planner.next_searches(room)
        turn = self.session.turn
        self.answer(said, beside=work)
        if self.session.turn == turn:
            return False
        self.order.unanswered = []
        self.reader.pending_line = ""
        self.reader.hear()
        return True
    def pick_stand_in(self, force: bool = False) -> Optional[Key]:
        ranked = [k for k in self.sieve.ranked() if k not in self.sieve.gone]
        best = ranked[0] if ranked else None
        best_title = Text.lowered((self.catalog.pages.get(best[0]) or {}).get("title")) if best else ""
        def other(key: Key) -> bool:
            return key != best and Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title")) != best_title
        fitting = [k for k in ranked if other(k) and self.sieve.meets_opening(k)]
        if fitting:
            more_is_better = self.sieve.better != "lower" and not re.search(r"\b(low|least|small|short|few|min|light|cheap)", Text.lowered(self.order.priority))
            def worth(key: Key) -> Tuple[Any, ...]:
                value, price = self.sieve.figure(key), (self.catalog.stock.get(key) or {}).get("price") or 0.0
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return (0, float(value) if more_is_better else -float(value), price)
                return (1, price if more_is_better else -price, 0.0)
            surest = max(self.sieve.trust(k) for k in fitting)
            chosen = min([k for k in fitting if self.sieve.trust(k) >= surest], key=lambda k: (self.sieve.doubts(k), -self.sieve.own_exact(k)) + worth(k))
            ordered = [chosen] + sorted([k for k in fitting if k != chosen], key=lambda k: (-self.sieve.trust(k), self.sieve.doubts(k)) + worth(k))
            picked, none_fit = self.decide(ordered, first=True)
            if picked is not None:
                return picked
            if not none_fit:
                return chosen
            if not force:
                return None
            return self.due_choice(ordered)
        if best is not None and self.sieve.meets_opening(best) and not self.order.changed_early:
            return best
        spare = [k for k in self.judge.verdicts if other(k) and k not in ranked and self.sieve.affordable(k) and self.sieve.meets_opening(k)]
        def checked(keys: List[Key]) -> Optional[Key]:
            middle = Text.middle([(self.catalog.stock.get(k) or {}).get("price") or 0.0 for k in keys])
            def plainness(key: Key) -> Tuple[Any, ...]:
                price = (self.catalog.stock.get(key) or {}).get("price") or 0.0
                return (-self.sieve.doubts(key), self.sieve.own_exact(key), sum(self.matcher.plain(key, r) + self.matcher.exact(key, r) for r in self.order.opening),
                        -abs(price - middle), -price)
            ordered = sorted(keys, key=plainness, reverse=True)
            picked, none_fit = self.decide(ordered, first=True)
            if picked is not None or not none_fit:
                return picked or ordered[0]
            return self.due_choice(ordered) if force else None
        if spare:
            found = checked(spare)
            if found is not None or not force:
                return found
        loose = [k for k in self.judge.verdicts if other(k) and self.sieve.affordable(k) and bool((self.judge.verdicts.get(k) or {}).get("kind_ok"))
                 and self.order.opening and all(self.sieve.reliance(k, r) >= LISTED_FIT for r in self.order.opening)]
        if not loose and force:
            loose = [k for k in self.judge.verdicts if self.sieve.affordable(k) and bool((self.judge.verdicts.get(k) or {}).get("kind_ok"))
                     and k not in self.sieve.gone and (not self.order.opening or all(self.sieve.reliance(k, r) > 0 for r in self.order.opening))]
            loose = sorted(loose, key=lambda k: (k not in ranked, -self.sieve.trust(k)))[:DECIDE_CARDS * 2]
        return checked(loose) if loose else None
    def cart_stand_in(self, force: bool = False) -> None:
        if self.catalog.cart or self.session.over or self.session.turns_left() < 9 or self.clock.late() or not self.session.has("add_to_cart"):
            return
        self.judge.settle()
        self.judge.rank_by_priority(every=True)
        key = self.pick_stand_in(force or self.first_cart_due())
        if key is None:
            return
        group = [StoreSession.call("add_to_cart", product_id=key[0], sku=key[1])]
        if self.session.has("inspect_stock"):
            group.append(StoreSession.call("inspect_stock", product_id=key[0], sku=key[1]))
        self.session.send(group)
        if key not in self.catalog.cart:
            return
        self.stand_in = key
        seen = self.catalog.stock.get(key) or {}
        price = seen.get("price")
        changed, _ = self.reader.hear()
        if seen.get("in_stock") is False or (self.order.budget is not None and isinstance(price, float) and price > self.order.budget + 1e-9):
            self.sieve.gone.add(key)
            self.lost_early(key)
            self.reader.pending_line = ""
        if changed and self.session.turns_left() > TURN_RESERVE and not self.clock.late():
            self.judge.settle()
            self.after_change_searches()
            fresh = self.planner.next_searches(self.session.group_size)
            if fresh:
                self.session.send(fresh)
                self.reader.hear_quietly()
            opens = self.planner.to_open(self.session.group_size * OPEN_BATCH, allowance=len(self.catalog.pages) + 60)
            if opens and not self.session.over:
                self.session.send(opens[: self.session.group_size])
                self.reader.hear_quietly()
            self.judge.settle()
            self.look(rounds=4)
    def hunt_best(self) -> None:
        if not self.order.priority or self.sieve.priced_priority() or not self.session.has("search"):
            return
        best_so_far: Optional[float] = None
        known_so_far = 0
        for round_number in range(BEST_ROUNDS):
            self.cart_due()
            chosen = [k for k in self.catalog.cart if k != self.stand_in]
            if self.session.over or chosen or self.clock.late() or not self.desk.can_think() or self.session.turns_left() <= TURN_RESERVE + 4:
                break
            self.judge.rank_by_priority(every=True)
            found = self.sieve.best_figures()
            best = found[0][0] if found else None
            better = best is not None and (best_so_far is None or (best < best_so_far if self.sieve.better == "lower" else best > best_so_far))
            if round_number and not better and (len(found) <= known_so_far or len(found) >= 8) and (len(found) >= 3 or round_number >= 3):
                break
            best_so_far, known_so_far = best if best is not None else best_so_far, len(found)
            shown = "\n".join(f"- {value:g}: {str((self.catalog.pages.get(key[0]) or {}).get('title') or '')[:90]} | {self.catalog.variant_of(key).get('options') or '-'}"
                              for value, key in found[:6]) or "(no fitting listing states a figure yet)"
            text = (f"{self.order.state_text()}\n\nBest figures found so far for the priority, with the listings that carry them:\n{shown}"
                    f"\n\nSearches tried:\n" + "\n".join(self.planner.searched[-40:]))
            lists = [[q for q in self.desk.listed_fields(answer)["queries"] if isinstance(q, str) and q.strip()][:8]
                     for answer in self.desk.ask_all(SEAT_BEST, text)]
            taken = {Text.word_set(q) for q in list(self.planner.sent) + self.planner.planned}
            queries = []
            for place in range(8):
                for one in lists:
                    query = Text.lowered(" ".join(one[place].split())[:QUERY_LENGTH]) if place < len(one) else ""
                    if Text.word_set(query) and Text.word_set(query) not in taken:
                        taken.add(Text.word_set(query))
                        queries.append(query)
            siblings = self.sibling_searches([k for _, k in found] + [k for k in self.sieve.ranked() if k not in self.sieve.gone], self.session.group_size // 2)
            queries = siblings + queries
            if not queries:
                break
            self.planner.planned = queries + self.planner.planned
            searches = self.planner.next_searches(self.session.group_size)
            if not searches:
                break
            self.session.send(searches)
            self.reader.hear_quietly()
            opens = self.planner.to_open(self.session.group_size * OPEN_BATCH, allowance=len(self.catalog.pages) + 60)
            if opens and not self.session.over and self.session.turns_left() > TURN_RESERVE + 2:
                self.session.send(opens[: self.session.group_size])
                self.reader.hear_quietly()
            self.judge.settle()
        self.judge.rank_by_priority(every=True)
    def sibling_searches(self, keys: List[Key], room: int) -> List[str]:
        every = " ".join(self.planner.values())
        out: List[str] = []
        for key in keys:
            title = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title"))
            if title and title not in self.planner.kin_swept and len(out) < room:
                self.planner.kin_swept.add(title)
                out.append(Text.lowered(f"{every} {' '.join(Text.word_list(title)[:12])}"[:QUERY_LENGTH]))
        return out
    def after_change_searches(self) -> None:
        near = [k for k in self.sieve.ranked() if k not in self.sieve.gone] + [k for k in self.sieve.near_misses() if k not in self.sieve.gone][:6]
        found = self.sibling_searches(list(dict.fromkeys(near)), max(1, self.session.group_size // 2))
        amounts = [q for q in self.planner.new_amount_queries() if Text.lowered(q) not in self.planner.sent]
        found = list(dict.fromkeys(found[:2] + [Text.lowered(q) for q in amounts] + found[2:]))
        if found:
            self.planner.planned = found + self.planner.planned
    def sure_choice(self) -> bool:
        ranked = [k for k in self.sieve.ranked() if k not in self.sieve.gone]
        if not ranked:
            return False
        best = ranked[0]
        if any(self.sieve.reliance(best, r) < FIRM_FIT or self.matcher.gainsaid(best, r) for r in self.order.requirements):
            return False
        if self.order.priority and not self.sieve.priced_priority():
            if self.sieve.figure(best) is None:
                return False
            if len(self.sieve.best_figures()) < 3 and self.session.turn < self.session.turn_limit // 2:
                return False
        return True
    def dig_until_sure(self) -> None:
        for round_number in range(10):
            self.cart_due()
            if self.session.over or self.session.turns_left() <= TURN_RESERVE + 2 or self.clock.late() or self.sure_choice():
                return
            before = (len(self.catalog.pages), len(self.planner.sent))
            self.planner.search_room = max(self.planner.search_room, len(self.planner.sent) + 12)
            self.planner.open_room = max(self.planner.open_room, len(self.catalog.pages) + 30)
            ways = (self.sweep, self.planner.rethink_searches) if round_number % 2 else (self.planner.rethink_searches, self.sweep)
            if not ways[0]() and not ways[1]():
                self.planner.kin_done.clear()
                self.planner.add_queries(self.planner.kin_queries() + self.planner.value_queries() + self.planner.shelf_queries(6), first=True)
                self.look(rounds=2, enough=99)
            if self.order.priority:
                self.judge.rank_by_priority(every=True)
            if (len(self.catalog.pages), len(self.planner.sent)) == before:
                return
            self.sieve.ranked()
    def recart(self) -> Optional[Key]:
        for key in [k for k in self.sieve.ranked() if k not in self.sieve.gone][:2]:
            if self.session.over or self.session.turns_left() < 3:
                break
            group = [StoreSession.call("add_to_cart", product_id=key[0], sku=key[1])]
            if self.session.has("inspect_stock"):
                group.append(StoreSession.call("inspect_stock", product_id=key[0], sku=key[1]))
            self.session.send(group)
            self.reader.hear()
            seen = self.catalog.stock.get(key) or {}
            if seen.get("in_stock") is not False and (self.order.budget is None or not isinstance(seen.get("price"), float) or seen["price"] <= self.order.budget + 1e-9):
                return key
            self.sieve.gone.add(key)
        return None
    def reply_before_paying(self) -> None:
        lines, self.order.unanswered = self.order.unanswered[-4:], []
        self.answer("\n".join(f"- {line}" for line in lines))
    def last_resort(self) -> None:
        if self.session.over or self.session.turns_left() < 1 or not self.session.has("place_test_order"):
            return
        alive = [k for k in self.catalog.cart if k not in self.sieve.gone]
        key = alive[-1] if alive else next((k for k in self.sieve.ranked() if k not in self.sieve.gone), None)
        if key is None:
            return
        if key not in self.catalog.cart:
            if self.session.turns_left() < 2:
                return
            self.session.send([StoreSession.call("add_to_cart", product_id=key[0], sku=key[1])])
        self.session.send([self.final_action(key)])
    def look(self, rounds: int, enough: int = 2) -> None:
        for round_number in range(rounds):
            if self.session.over or self.session.turns_left() <= TURN_RESERVE or self.clock.late():
                break
            self.cart_due()
            opens = self.planner.to_open(self.session.group_size * OPEN_BATCH)[: self.session.group_size]
            ride: List[Dict[str, Any]] = []
            if opens and round_number and len(opens) < self.session.group_size:
                fitting = [k for k in self.sieve.ranked() if self.sieve.solid(k)]
                if len([k for k in fitting if self.sieve.firm(k)]) < enough:
                    ride = self.planner.round_searches(self.session.group_size - len(opens), fitting, enough)
            if opens or ride:
                self.session.send(opens + ride)
                self.reader.hear_quietly()
            self.judge.settle()
            fitting = [k for k in self.sieve.ranked() if self.sieve.solid(k)]
            if len([k for k in fitting if self.sieve.firm(k)]) >= enough:
                return
            if len(fitting) >= enough and not self.planner.kin_pending() and round_number >= 2:
                return
            if ride:
                continue
            searches = self.planner.round_searches(self.session.group_size, fitting, enough)
            if not searches and not opens:
                break
            if searches:
                self.session.send(searches)
                self.reader.hear_quietly()
        opens = self.planner.to_open(self.session.group_size * OPEN_BATCH)
        if opens and not self.session.over and self.session.turns_left() > TURN_RESERVE:
            self.session.send(opens[: self.session.group_size])
            self.reader.hear_quietly()
        self.judge.settle()
        if len(self.sieve.ranked()) < enough and not self.planner.widened and not self.session.over:
            self.planner.widen()
            return self.look(rounds=3, enough=enough)
        for _ in range(2):
            if len(self.sieve.ranked()) >= enough:
                return
            self.judge.ask_reader()
        if len(self.sieve.ranked()) < enough and not self.matcher.lenient:
            self.matcher.lenient = True
            self.judge.settle()
    def refill(self, lost: Key, wide: bool = True) -> None:
        words = Text.word_list((self.catalog.pages.get(lost[0]) or {}).get("title"))
        if len(words) < 2 or not self.session.has("search"):
            return
        every = " ".join(self.planner.values())
        own = " ".join(w for w in Text.word_list(self.catalog.variant_of(lost).get("options")) if not w.replace(".", "").isdigit())
        queries = [" ".join(words), " ".join(words[:10])]
        if own:
            queries.append(f"{own} {' '.join(words[:10])}")
        queries += [f"{every} {' '.join(words[:10])}", f"{every} {' '.join(words[:5])}", f"{every} {self.order.item} {' '.join(words[:3])}"]
        if wide:
            queries += [f"{every} {' '.join(words)}"] + self.planner.shelf_queries(6)
        texts = list(dict.fromkeys(Text.lowered(" ".join(q.split())[:QUERY_LENGTH]) for q in queries))
        calls = [StoreSession.call("search", query=" ".join(text.replace('"', " ").split()), k=10, in_stock=True) for text in texts]
        label, price = self.planner.store_label(), (self.catalog.stock.get(lost) or {}).get("price")
        if wide and label and isinstance(price, float) and self.session.has("filter"):
            ceiling = round(min(self.order.budget or price * 1.1, price * 1.1), 2)
            brand = str((self.catalog.pages.get(lost[0]) or {}).get("brand") or "").strip()
            calls.append(StoreSession.call("filter", category=label, max_price=ceiling, in_stock=True, k=10))
            if brand and Text.lowered(brand) not in NO_MAKER:
                calls.append(StoreSession.call("filter", category=label, brand=brand, max_price=ceiling, in_stock=True, k=10))
        self.session.send(calls[: self.session.group_size])
        self.reader.hear_quietly()
        opens = self.planner.to_open(4 * OPEN_BATCH, allowance=len(self.catalog.pages) + 4 * OPEN_BATCH)
        if opens and not self.session.over:
            self.session.send(opens)
            self.reader.hear_quietly()
        self.judge.settle()
    def places(self, key: Key) -> Dict[str, str]:
        page = self.catalog.pages.get(key[0]) or {}
        places = {"title": str(page.get("title") or ""), "options": str(self.catalog.variant_of(key).get("options") or "")}
        places.update({str(label): str(value) for label, value in (page.get("facts") or {}).items()})
        return places
    def stated_at(self, key: Key, requirement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        places = self.places(key)
        page = self.catalog.pages.get(key[0]) or {}
        own = [label for label, _ in self.matcher.named_facts(page, requirement)]
        order = own + ["options", "title"] + [label for label in places if label not in own and label not in ("options", "title")]
        forms = sorted(self.matcher.forms(requirement) | {" ".join(Text.word_list(Text.closed_up(requirement["value"])))}, key=len, reverse=True)
        for field in order:
            text = places.get(field) or ""
            for reading in (Text.lowered(text), Text.closed_up(text)):
                for form in forms:
                    if not form:
                        continue
                    found = re.search(r"(?<![a-z0-9])" + re.escape(form).replace(r"\ ", r"\W+") + r"(?![a-z0-9])", reading)
                    if found and f" {' '.join(Text.word_list(found.group(0)))} " in f" {' '.join(Text.word_list(text))} ":
                        return {"field": field, "value": found.group(0)}
            words = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())
            for form in forms:
                wanted = form.split()
                for start in range(len(words) - len(wanted) + 1):
                    if wanted and all(Text.singular(words[start + i]) == w or Text.same_word(words[start + i], w) for i, w in enumerate(wanted)):
                        return {"field": field, "value": " ".join(words[start:start + len(wanted)])}
        return None
    def claims(self, key: Key) -> Optional[Dict[str, Any]]:
        seen = self.catalog.stock.get(key) or {}
        places = self.places(key)
        claims: List[Dict[str, Any]] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": seen.get("in_stock") is not False})
        fact = None
        verdict = self.judge.verdicts.get(key) or {}
        asked = Text.name_words(self.order.backed_by) - {"type", "kind", "item", "detail"}
        about: Optional[Dict[str, Any]] = None
        if asked:
            about = next((r for r in reversed(self.order.requirements) if Text.name_words(r["property"]) & asked), None)
        ordered = ([about] if about else []) + [r for r in self.order.requirements if r.get("stated")] + [r for r in self.order.requirements if not r.get("stated")]
        if asked:
            fact = (self.stated_at(key, about) if about else None) or self.named_place(key, asked)
            if fact is None:
                for requirement in ordered:
                    fact = self.stated_at(key, requirement) or self.named_place(key, Text.name_words(requirement["property"]) - {"type", "kind", "item", "detail"})
                    if fact:
                        break
        for requirement in ([] if asked else ordered):
            fact = self.stated_at(key, requirement)
            if fact:
                break
        if fact is None:
            for where, words in zip(verdict.get("where") or [], verdict.get("words") or []):
                where = re.sub(r"^facts?\s*[:.]\s*", "", str(where or ""))
                field = next((label for label in places if Text.lowered(label) == Text.lowered(where)), None)
                said = " ".join(Text.word_list(words))
                if field and said and f" {said} " in f" {' '.join(Text.word_list(places[field]))} ":
                    fact = {"field": field, "value": str(words).strip()}
                    break
        if fact:
            claims.append(fact)
        return {"claims": claims}
    def named_place(self, key: Key, asked: Set[str]) -> Optional[Dict[str, Any]]:
        places = self.places(key)
        for label, value in places.items():
            if label not in ("title", "options") and Text.name_words(label) & asked and Text.word_list(value):
                return {"field": label, "value": str(value).strip()}
        for field in ("title", "options"):
            words = Text.word_list(places.get(field))
            at = next((i for i, w in enumerate(words) if Text.singular(w) in asked), None)
            if at is not None:
                return {"field": field, "value": " ".join(words[max(0, at - 2): at + 1])}
        return None
    def shortfall(self, key: Key) -> Optional[Dict[str, Any]]:
        seen = self.catalog.stock.get(key) or {}
        page = self.catalog.pages.get(key[0]) or {}
        places = self.places(key)
        claims: List[Dict[str, Any]] = []
        if isinstance(seen.get("price"), float):
            claims.append({"field": "price", "value": seen["price"]})
        claims.append({"field": "in_stock", "value": seen.get("in_stock") is not False})
        missed = sorted((r for r in self.order.requirements if self.matcher.plain(key, r) < 1), key=lambda r: self.matcher.plain(key, r)) or list(self.order.requirements)
        for requirement in missed:
            for label, value in self.matcher.named_facts(page, requirement):
                words = " ".join(Text.word_list(value))
                if words:
                    claims.append({"field": label, "value": str(value).strip()})
                    return {"claims": claims}
        for field in ("options", "title"):
            if " ".join(Text.word_list(places.get(field))):
                claims.append({"field": field, "value": places[field].strip()})
                break
        return {"claims": claims}
    def look_again(self, choice: Key) -> None:
        stamp = json.dumps([[(r["property"], r["value"]) for r in self.order.requirements], self.order.budget], sort_keys=True)
        if not self.order.look_at or self.looked_under == stamp or self.session.over or self.session.turns_left() < 2 or not (self.session.has("compare") or self.session.has("view")):
            return
        self.looked_under = stamp
        wanted = self.order.look_at + 2
        picked: List[str] = []
        titles: Set[str] = set()
        for key in [choice] + [k for k in self.sieve.ranked() if k not in self.sieve.gone] + self.sieve.near_misses():
            title = Text.lowered((self.catalog.pages.get(key[0]) or {}).get("title"))
            if key[0] in picked or (title in titles and key != choice) or not self.sieve.on_sale(key):
                continue
            picked.append(key[0])
            titles.add(title)
            if len(picked) >= wanted:
                break
        if self.session.has("compare"):
            calls = [StoreSession.call("compare", product_ids=picked[i:i + OPEN_BATCH]) for i in range(0, len(picked), OPEN_BATCH)]
        else:
            calls = [StoreSession.call("view", product_id=i) for i in picked]
        self.session.send(calls)
        self.reader.hear_quietly()
    def final_action(self, choice: Key) -> Dict[str, Any]:
        if choice == self.named_nearest and self.can_abstain():
            return StoreSession.call("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.claims(choice))
        if choice in self.catalog.cart or not self.can_abstain():
            return StoreSession.call("place_test_order", product_id=choice[0], sku=choice[1], justification=self.claims(choice))
        return StoreSession.call("place_test_order", product_id=choice[0], sku=choice[1], abstain=True, justification=self.shortfall(choice))
    def nearest_now(self, key: Key, read: bool = True) -> Optional[Key]:
        if read and self.session.has("inspect_stock") and self.session.turns_left() >= 2:
            self.session.send([StoreSession.call("inspect_stock", product_id=key[0], sku=key[1])])
            if self.catalog.heard and self.reader.hear()[0]:
                self.judge.settle()
        left = [k for k in self.sieve.ranked() if k not in self.sieve.gone]
        if left and max(self.sieve.trust(k) for k in left[:12]) < LIKELY_FIT and any(self.sieve.trust(k) >= SURE_FIT for k in self.sieve.only_over()):
            left = []
        if left:
            self.named_nearest = None
            if left[0] not in self.catalog.cart and self.session.turns_left() >= 2:
                self.session.send([StoreSession.call("add_to_cart", product_id=left[0][0], sku=left[0][1])])
                self.reader.hear_quietly()
            if left[0] in self.catalog.cart:
                return left[0]
            alive = [k for k in self.catalog.cart if k not in self.sieve.gone]
            return alive[-1] if alive else self.sieve.closest()
        over = self.sieve.only_over()
        if over:
            self.named_nearest = key if key in over else over[0]
            return self.named_nearest
        self.named_nearest = None
        alive = [k for k in self.catalog.cart if k not in self.sieve.gone]
        return alive[-1] if alive else self.sieve.closest()
    def can_abstain(self) -> bool:
        spec = (self.session.tools.get("place_test_order") or {}).get("parameters") or {}
        return "abstain" in (spec.get("properties") or {})
    def answer(self, line: str, beside: Optional[List[Dict[str, Any]]] = None) -> None:
        if self.session.over or not self.session.has("message") or self.session.turns_left() <= 2:
            return
        if line in self.order.unanswered:
            self.order.unanswered.remove(line)
        news, _ = self.order.holds_now(line)
        if not news and re.search(r"\bconfirm|\bright\?|\bok(?:ay)?\?", Text.lowered(line)):
            news = [r for r in self.order.requirements if Text.says_value(line, r["value"])]
        if not news:
            asked = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n", line) if part.strip().endswith("?")]
            if asked:
                self.session.send([StoreSession.call("message", content="Yes, confirmed: it stays as you said. Everything else stays the same, your budget included.")]
                             + list(beside or []))
            return
        parts = []
        for requirement in news:
            before = self.order.took_back.get(Text.lowered(requirement["value"]))
            if requirement.get("also") and not before:
                parts.append(f"the {requirement['property']} has to include {requirement['value']} as well as {requirement['also']}")
            else:
                parts.append(f"the {requirement['property']} is now {requirement['value']}" + (f" instead of {before}" if before else ""))
        self.session.send([StoreSession.call("message", content=("Yes, confirmed: " + ", and ".join(parts) + ". Everything else stays the same, your budget included.")[:400])]
                     + list(beside or []))
        self.reader.reply_sent = (self.session.turn, line)
    def run(self) -> List[Dict[str, Any]]:
        question = self.reader.read_request()
        self.order.opening, self.order.opening_budget = [dict(r) for r in self.order.requirements if r.get("stated")], self.order.budget
        first: List[Dict[str, Any]] = []
        if question and self.session.has("message"):
            first.append(self.interviewer.put(question))
        if self.order.listed and self.session.has("compare"):
            first.append(StoreSession.call("compare", product_ids=list(dict.fromkeys(k[0] for k in self.order.listed))[:OPEN_BATCH]))
        filters = [StoreSession.call("filter", category=guess, max_price=self.order.budget, in_stock=True, k=10) for guess in self.planner.label_guesses()[:5]] if self.session.has("filter") else []
        first.extend(self.planner.next_searches(self.session.group_size - len(first) - len(filters)))
        self.session.send(first + filters)
        self.interviewer.listen()
        if self.interviewer.pair_again and not self.order.pair_chosen and self.session.has("message") and not self.session.over:
            again = [self.interviewer.put(self.order.pair_question)] + self.planner.next_searches(self.session.group_size - 1)
            self.session.send(again)
            self.interviewer.listen()
        if self.order.pair_fallback and not self.order.pair_chosen:
            self.order.requirements += self.order.set_aside
            self.order.opening += [dict(r) for r in self.order.set_aside if r.get("stated")]
            self.order.brand = self.order.brand or self.order.pair_brand
            self.order.set_aside, self.order.pair_fallback = [], ""
            self.interviewer.plain_pending = self.order.invited or self.order.told_to_ask
            self.planner.add_queries(self.planner.value_queries(), first=True)
        try:
            self.interviewer.find_out()
        except Exception:
            self.interviewer.awaiting_answer = False
        try:
            self.settle_early()
        except Exception:
            pass
        self.look(rounds=9)
        try:
            if self.read_store_words() and self.session.turns_left() > TURN_RESERVE + 6:
                self.look(rounds=2)
        except Exception:
            pass
        try:
            self.cart_stand_in()
        except Exception:
            pass
        try:
            self.hunt_best()
        except Exception:
            pass
        try:
            self.dig_until_sure()
        except Exception:
            pass
        self.cart_due()
        commits = 0
        choice: Optional[Key] = None
        while not self.session.over and self.session.turns_left() >= 2 and commits < 5:
            held = self.best_held() if self.clock.overdue() else None
            if held is not None and self.can_abstain() and not any(self.sieve.trust(k) >= LIKELY_FIT for k in self.sieve.ranked() if k not in self.sieve.gone):
                over = [k for k in self.sieve.only_over() if self.sieve.trust(k) >= SURE_FIT]
                if over:
                    choice = self.named_nearest = over[0]
                    break
            if held is not None:
                choice = held
                break
            self.judge.rank_by_priority(every=True)
            open_keys = [k for k in self.sieve.ranked() if k not in self.sieve.gone]
            weak = (bool(open_keys) and not self.planner.search_asks and self.session.turns_left() >= 14
                    and max(self.sieve.trust(k) for k in open_keys[:12]) < LIKELY_FIT)
            if (not open_keys or weak or not self.catalog.cart and not any(self.sieve.keeps_request(k) for k in open_keys)) and self.planner.rethink_searches():
                continue
            if (not open_keys or max(self.sieve.trust(k) for k in open_keys[:12]) < SURE_FIT) and self.sweep():
                continue
            loosely = bool(open_keys) and max(self.sieve.trust(k) for k in open_keys[:12]) < LIKELY_FIT
            if (not open_keys or loosely) and self.can_abstain():
                over = [k for k in self.sieve.only_over() if not open_keys or self.sieve.trust(k) >= SURE_FIT]
                if over:
                    choice = self.named_nearest = over[0]
                    break
            order = open_keys or [k for k in self.sieve.near_misses() if k not in self.sieve.gone]
            if not order:
                break
            if not self.catalog.cart:
                order.sort(key=lambda k: (not self.sieve.meets_chosen(k), not self.sieve.keeps_request(k)))
            pool = order[: DECIDE_CARDS - 2] + [k for k in self.sieve.near_misses() if k not in order and k not in self.sieve.gone][:2]
            record_before = json.dumps(self.order.requirements, sort_keys=True, default=str)
            picked, none_fit = self.decide(pool)
            if json.dumps(self.order.requirements, sort_keys=True, default=str) != record_before and self.record_repairs < 2:
                self.record_repairs += 1
                self.judge.settle()
                if self.session.turns_left() > TURN_RESERVE and not self.clock.late():
                    self.planner.add_queries(self.planner.value_queries(), first=True)
                    self.look(rounds=2, enough=1)
                continue
            if picked is not None:
                order = [picked] + [k for k in order if k != picked]
            elif none_fit and not self.decide_hunted and self.session.turns_left() > TURN_RESERVE + 2 and not self.clock.late():
                self.decide_hunted = True
                if not self.sweep(force=True) and self.planner.rethink_searches("none of the candidates meets every requirement the shopper's words set"):
                    self.look(rounds=3, enough=1)
                continue
            if not open_keys and self.near_opens < 2 and self.session.turns_left() > TURN_RESERVE + 1 and not self.clock.late():
                opens = self.planner.to_open(self.session.group_size * OPEN_BATCH, allowance=len(self.catalog.pages) + self.session.group_size * OPEN_BATCH)[: self.session.group_size]
                if opens:
                    self.near_opens += 1
                    self.session.send(opens)
                    self.reader.hear_quietly()
                    self.judge.settle()
                    continue
            if self.late_opens < 2 and self.session.turns_left() > TURN_RESERVE + 1 and not self.clock.late():
                rows = self.planner.rows_for(order[0])
                if rows:
                    self.late_opens += 1
                    self.session.send(rows)
                    self.reader.hear_quietly()
                    self.judge.settle()
                    continue
            choice = order[0]
            commits += 1
            self.sieve.ranked()
            group = []
            if choice not in self.catalog.cart:
                group.append(StoreSession.call("add_to_cart", product_id=choice[0], sku=choice[1]))
            if self.session.has("inspect_stock"):
                group.append(StoreSession.call("inspect_stock", product_id=choice[0], sku=choice[1]))
            if self.stand_in in self.catalog.cart and self.stand_in != choice and self.session.has("remove_from_cart"):
                group.append(StoreSession.call("remove_from_cart", product_id=self.stand_in[0], sku=self.stand_in[1]))
                self.catalog.cart.remove(self.stand_in)
            self.session.send(group)
            seen = self.catalog.stock.get(choice) or {}
            price = seen.get("price")
            over_budget = self.order.budget is not None and isinstance(price, float) and price > self.order.budget + 1e-9
            changed, asks = self.reader.hear()
            if seen.get("in_stock") is False or over_budget:
                self.sieve.gone.add(choice)
                self.sieve.lost_titles.add(Text.lowered((self.catalog.pages.get(choice[0]) or {}).get("title")))
                self.reader.pending_line = ""
                if self.session.turns_left() > TURN_RESERVE and not self.clock.late():
                    self.refill(choice, wide=not any(self.sieve.firm(k) for k in self.sieve.ranked() if k not in self.sieve.gone))
                left = [k for k in self.sieve.ranked() if k not in self.sieve.gone]
                if len([k for k in left if self.sieve.firm(k)]) < 2 and self.session.turns_left() > TURN_RESERVE and not self.clock.late():
                    self.planner.kin_done.clear()
                    self.planner.add_queries(self.planner.kin_queries() + self.planner.shelf_queries(6), first=True)
                    self.look(rounds=3 if left else 5, enough=2 if left else 1)
                choice = None
                continue
            if changed:
                self.judge.settle()
                try:
                    self.read_store_words()
                except Exception:
                    pass
                if not self.sieve.fits(choice):
                    self.sieve.gone.add(choice)
                    if self.session.turns_left() > TURN_RESERVE and not self.clock.late():
                        self.after_change_searches()
                        fresh = self.planner.next_searches(self.session.group_size)
                        if fresh:
                            self.session.send(fresh)
                            self.reader.hear_quietly()
                        opens = self.planner.to_open(self.session.group_size * OPEN_BATCH, allowance=len(self.catalog.pages) + 60)
                        if opens and not self.session.over:
                            self.session.send(opens[: self.session.group_size])
                            self.reader.hear_quietly()
                        self.judge.settle()
                    if not [k for k in self.sieve.ranked() if k not in self.sieve.gone] and self.session.turns_left() > TURN_RESERVE and not self.clock.late():
                        self.refill(choice, wide=False)
                    if not [k for k in self.sieve.ranked() if k not in self.sieve.gone] and self.session.turns_left() > TURN_RESERVE and not self.clock.late():
                        self.look(rounds=4, enough=1)
                    choice = None
                    continue
            if asks and self.reader.pending_line:
                self.answer(self.reader.pending_line)
                self.reader.pending_line = ""
                more_changed, _ = self.reader.hear()
                if more_changed:
                    continue
            if self.session.turn > int((self.catalog.stock.get(choice) or {}).get("turn") or 0) and self.session.has("inspect_stock") and self.session.turns_left() >= 3:
                self.session.send([StoreSession.call("inspect_stock", product_id=choice[0], sku=choice[1])])
                late_change, _ = self.reader.hear()
                seen = self.catalog.stock.get(choice) or {}
                if seen.get("in_stock") is False or (self.order.budget is not None and isinstance(seen.get("price"), float) and seen["price"] > self.order.budget + 1e-9):
                    self.sieve.gone.add(choice)
                    self.sieve.lost_titles.add(Text.lowered((self.catalog.pages.get(choice[0]) or {}).get("title")))
                    choice = None
                    continue
                if late_change:
                    self.judge.settle()
                    if not self.sieve.fits(choice):
                        self.sieve.gone.add(choice)
                        choice = None
                        continue
            if self.order.unanswered and self.session.has("message") and self.session.turns_left() >= 3:
                self.reply_before_paying()
                if self.reader.hear()[0]:
                    self.judge.settle()
                    if not self.sieve.fits(choice):
                        self.sieve.gone.add(choice)
                        choice = None
                        continue
            break
        if choice is None and not self.session.over and self.can_abstain() and not [k for k in self.sieve.ranked() if k not in self.sieve.gone]:
            over = self.sieve.only_over()
            if over:
                choice = self.named_nearest = over[0]
        if choice is None and self.catalog.cart and not self.session.over:
            alive = [k for k in self.catalog.cart if k not in self.sieve.gone]
            choice = alive[-1] if alive else None
        if choice is None and not self.session.over:
            self.judge.settle()
            nearest = self.sieve.closest()
            if nearest is not None and nearest not in self.catalog.cart and self.session.turns_left() >= 2:
                self.session.send([StoreSession.call("add_to_cart", product_id=nearest[0], sku=nearest[1])])
                self.reader.hear_quietly()
            if nearest is not None and (nearest in self.catalog.cart or self.can_abstain()):
                choice = nearest
        for _ in range(3):
            if choice is None or self.session.over or self.session.turns_left() < 1:
                break
            if self.order.unanswered and self.session.has("message") and self.session.turns_left() >= 2:
                self.reply_before_paying()
            if choice == self.named_nearest:
                choice = self.nearest_now(choice)
                if choice is None:
                    break
            elif not self.clock.overdue():
                self.look_again(choice)
            if self.catalog.heard:
                if self.reader.hear()[0]:
                    self.judge.settle()
                    if choice == self.named_nearest:
                        choice = self.nearest_now(choice, read=False)
                        if choice is None:
                            break
                    elif not self.sieve.fits(choice):
                        self.sieve.gone.add(choice)
                        choice = self.recart() or choice
            observations = self.session.send([self.final_action(choice)])
            if self.session.over:
                break
            changed, asks = self.reader.hear()
            if asks and self.reader.pending_line and self.session.turns_left() >= 2:
                self.answer(self.reader.pending_line)
                self.reader.pending_line = ""
                changed = self.reader.hear()[0] or changed
            if changed and self.session.turns_left() >= 3:
                self.judge.settle()
                if choice == self.named_nearest:
                    choice = self.nearest_now(choice, read=False)
                elif not self.sieve.fits(choice):
                    self.sieve.gone.add(choice)
                    choice = self.recart()
            if not observations:
                break
        return self.session.dialogue


def agent_main(problem_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    Text.memo.clear()
    episode = Episode(problem_data)
    try:
        return episode.run()
    except Exception:
        try:
            episode.last_resort()
        except Exception:
            pass
        return episode.session.dialogue


__all__ = ["agent_main"]
