import asyncio
import json
import logging
import time
from typing import Callable, Optional

import websockets

log = logging.getLogger("ws")

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
HEARTBEAT_INTERVAL = 10  # seconds
RECONNECT_DELAY = 5      # seconds


class PolymarketWS:
    """WebSocket client for real-time Polymarket price updates on open positions."""

    def __init__(self):
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscribed_tokens: set = set()
        self._token_to_market: dict[str, str] = {}
        # market_id -> latest data
        self.prices: dict[str, dict] = {}
        # callbacks
        self._on_price_change: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_disconnect: Optional[Callable] = None
        self._on_reconnect: Optional[Callable] = None

    def set_callbacks(self, on_price_change=None, on_trade=None,
                      on_disconnect=None, on_reconnect=None):
        self._on_price_change = on_price_change
        self._on_trade = on_trade
        self._on_disconnect = on_disconnect
        self._on_reconnect = on_reconnect

    def register_market(self, market_id: str, yes_token: str = None, no_token: str = None,
                        yes_price: float = 0.5, question: str = ""):
        """Register a single market for WS subscription (e.g. when opening a position)."""
        tokens_to_add = []
        if yes_token:
            self._token_to_market[yes_token] = market_id
            self._subscribed_tokens.add(yes_token)
            tokens_to_add.append(yes_token)
        if no_token:
            self._token_to_market[no_token] = market_id
            self._subscribed_tokens.add(no_token)
            tokens_to_add.append(no_token)
        if market_id not in self.prices:
            self.prices[market_id] = {
                "yes_price": yes_price,
                "question": question,
                "yes_token": yes_token,
                "no_token": no_token,
                "last_update": time.time(),
            }
        else:
            if yes_token and not self.prices[market_id].get("yes_token"):
                self.prices[market_id]["yes_token"] = yes_token
            if no_token and not self.prices[market_id].get("no_token"):
                self.prices[market_id]["no_token"] = no_token
        return tokens_to_add

    async def subscribe_market(self, market_id: str, yes_token: str = None, no_token: str = None,
                               yes_price: float = 0.5, question: str = ""):
        """Register and immediately subscribe to a market's tokens."""
        # Check which tokens are new BEFORE register_market adds them to the set
        new_tokens = []
        if yes_token and yes_token not in self._subscribed_tokens:
            new_tokens.append(yes_token)
        if no_token and no_token not in self._subscribed_tokens:
            new_tokens.append(no_token)
        self.register_market(market_id, yes_token, no_token, yes_price, question)
        if new_tokens and self.ws:
            await self._send_subscribe(new_tokens)
            log.info(f"[WS] Subscribed to {market_id[:8]} ({len(new_tokens)} tokens)")

    async def unsubscribe_market(self, market_id: str):
        """Unsubscribe from a market's tokens (e.g. when closing a position)."""
        info = self.prices.pop(market_id, None)
        if not info:
            return
        tokens_to_remove = []
        for token_field in ("yes_token", "no_token"):
            token = info.get(token_field)
            if token:
                self._token_to_market.pop(token, None)
                self._subscribed_tokens.discard(token)
                tokens_to_remove.append(token)
        if tokens_to_remove and self.ws:
            try:
                msg = {"assets_ids": tokens_to_remove, "type": "market", "action": "unsubscribe"}
                await self.ws.send(json.dumps(msg))
                log.info(f"[WS] Unsubscribed from {market_id[:8]} ({len(tokens_to_remove)} tokens)")
            except Exception:
                pass

    async def connect(self):
        """Connect to WebSocket and start listening."""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(WS_URL, ping_interval=None) as ws:
                    self.ws = ws
                    log.info(f"[WS] Connected, subscribing to {len(self._subscribed_tokens)} tokens")
                    if self._on_reconnect:
                        try: await self._on_reconnect()
                        except Exception: pass
                    await self._subscribe_all(ws)
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                    try:
                        async for message in ws:
                            if message == "PONG":
                                continue
                            try:
                                data = json.loads(message)
                                await self._handle_message(data)
                            except json.JSONDecodeError:
                                continue
                    finally:
                        heartbeat_task.cancel()
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                log.warning(f"[WS] Disconnected: {e}, reconnecting in {RECONNECT_DELAY}s")
                self.ws = None
                if self._on_disconnect:
                    try: await self._on_disconnect()
                    except Exception: pass
                await asyncio.sleep(RECONNECT_DELAY)
            except Exception as e:
                log.error(f"[WS] Error: {e}", exc_info=True)
                self.ws = None
                if self._on_disconnect:
                    try: await self._on_disconnect()
                    except Exception: pass
                await asyncio.sleep(RECONNECT_DELAY)

    @property
    def connected(self) -> bool:
        return self.ws is not None and self._running

    async def _subscribe_all(self, ws):
        """Send subscription for all tracked tokens in batches of 100."""
        if not self._subscribed_tokens:
            return
        tokens = list(self._subscribed_tokens)
        for i in range(0, len(tokens), 100):
            batch = tokens[i:i + 100]
            await self._send_subscribe(batch, ws)
            log.info(f"[WS] Subscribed batch {i // 100 + 1}: {len(batch)} tokens")

    async def _send_subscribe(self, token_ids: list, ws=None):
        ws = ws or self.ws
        if not ws:
            return
        msg = {"assets_ids": token_ids, "type": "market", "custom_feature_enabled": True}
        await ws.send(json.dumps(msg))

    async def _heartbeat(self, ws):
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL)
                await ws.send("PING")
            except Exception:
                break

    async def _handle_message(self, data):
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    await self._handle_message(item)
            return
        if not isinstance(data, dict):
            return
        event_type = data.get("event_type")
        if event_type == "price_change":
            await self._handle_price_change(data)
        elif event_type == "last_trade_price":
            await self._handle_trade(data)
        elif event_type == "book":
            await self._handle_book(data)

    async def _handle_price_change(self, data: dict):
        for change in data.get("price_changes", []):
            asset_id = change.get("asset_id")
            market_id = self._token_to_market.get(asset_id)
            if not market_id or market_id not in self.prices:
                continue
            best_bid = change.get("best_bid")
            best_ask = change.get("best_ask")
            if best_bid and best_ask:
                bid_f = float(best_bid)
                ask_f = float(best_ask)
                spread = round(ask_f - bid_f, 4)
                # Skip wide spreads — price is unreliable
                if ask_f > 0 and spread / ask_f > 0.20:
                    continue
                mid_price = (bid_f + ask_f) / 2
                is_yes = self.prices[market_id].get("yes_token") == asset_id
                yes_price = round(mid_price, 4) if is_yes else round(1 - mid_price, 4)
                old_price = self.prices[market_id].get("yes_price", 0)
                self.prices[market_id]["yes_price"] = yes_price
                self.prices[market_id]["best_bid"] = bid_f if is_yes else round(1 - ask_f, 4)
                self.prices[market_id]["best_ask"] = ask_f if is_yes else round(1 - bid_f, 4)
                self.prices[market_id]["spread"] = spread
                self.prices[market_id]["last_update"] = time.time()
                if self._on_price_change and abs(yes_price - old_price) > 0.0001:
                    await self._on_price_change(market_id, old_price, yes_price)

    async def _handle_trade(self, data: dict):
        asset_id = data.get("asset_id")
        market_id = self._token_to_market.get(asset_id)
        if not market_id or market_id not in self.prices:
            return
        price = float(data.get("price", 0))
        size = float(data.get("size", 0))
        side = data.get("side", "")
        is_yes = self.prices[market_id].get("yes_token") == asset_id
        yes_price = round(price, 4) if is_yes else round(1 - price, 4)
        old_price = self.prices[market_id].get("yes_price", 0)
        self.prices[market_id]["yes_price"] = yes_price
        self.prices[market_id]["last_trade_size"] = size
        self.prices[market_id]["last_trade_side"] = side
        self.prices[market_id]["last_update"] = time.time()
        # Trades are real executed prices — trigger SL/TP check
        if self._on_price_change and abs(yes_price - old_price) > 0.0001:
            await self._on_price_change(market_id, old_price, yes_price)
        if self._on_trade and size > 0:
            await self._on_trade(market_id, yes_price, size, side)

    async def _handle_book(self, data: dict):
        asset_id = data.get("asset_id")
        market_id = self._token_to_market.get(asset_id)
        if not market_id or market_id not in self.prices:
            return
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if bids and asks:
            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])
            spread = round(best_ask - best_bid, 4)
            # Skip wide spreads — price is unreliable
            if best_ask > 0 and spread / best_ask > 0.20:
                return
            mid = (best_bid + best_ask) / 2
            is_yes = self.prices[market_id].get("yes_token") == asset_id
            old_price = self.prices[market_id].get("yes_price", 0)
            yes_price = round(mid, 4) if is_yes else round(1 - mid, 4)
            self.prices[market_id]["yes_price"] = yes_price
            self.prices[market_id]["best_bid"] = best_bid if is_yes else round(1 - best_ask, 4)
            self.prices[market_id]["best_ask"] = best_ask if is_yes else round(1 - best_bid, 4)
            self.prices[market_id]["spread"] = spread
            self.prices[market_id]["last_update"] = time.time()
            # Trigger SL/TP check on meaningful book price changes
            if self._on_price_change and abs(yes_price - old_price) > 0.0001:
                await self._on_price_change(market_id, old_price, yes_price)

    def get_price(self, market_id: str) -> float:
        return self.prices.get(market_id, {}).get("yes_price", 0)

    def get_market_data(self, market_id: str) -> dict:
        return self.prices.get(market_id, {})

    def active_count(self) -> int:
        """Count markets with fresh data (updated in last 30s)."""
        cutoff = time.time() - 30
        return sum(1 for p in self.prices.values() if p.get("last_update", 0) > cutoff)

    def stop(self):
        self._running = False
