// WebSocket client — multiplexed message hub with auto-reconnect.
// Buffers messages that arrive before handlers are registered.
(function () {
  "use strict";

  var handlers = {};
  var buffer = [];  // Messages received before handlers registered
  var socket = null;
  var reconnectTimer = null;

  function getWsUrl() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + "/ws";
  }

  function dispatch(msg) {
    var type = msg.type;
    if (type && handlers[type]) {
      for (var i = 0; i < handlers[type].length; i++) {
        try {
          handlers[type][i](msg.data);
        } catch (e) {
          console.error("[ws] handler error for type '" + type + "':", e);
        }
      }
      return true;
    }
    return false;
  }

  function connect() {
    if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
      return;
    }

    socket = new WebSocket(getWsUrl());

    socket.onopen = function () {
      console.log("[ws] connected");
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    socket.onmessage = function (event) {
      var msg;
      try {
        msg = JSON.parse(event.data);
      } catch (e) {
        console.warn("[ws] invalid JSON:", event.data);
        return;
      }
      if (!dispatch(msg)) {
        // No handler yet — buffer for replay when handler registers
        buffer.push(msg);
      }
    };

    socket.onclose = function () {
      console.log("[ws] disconnected, reconnecting in 3s");
      scheduleReconnect();
    };

    socket.onerror = function () {
      // onclose will fire after onerror
    };
  }

  function scheduleReconnect() {
    if (!reconnectTimer) {
      reconnectTimer = setTimeout(function () {
        reconnectTimer = null;
        connect();
      }, 3000);
    }
  }

  /**
   * Register a handler for a given message type.
   * Replays any buffered messages of this type immediately.
   */
  window.wsOn = function (type, callback) {
    if (!handlers[type]) {
      handlers[type] = [];
    }
    handlers[type].push(callback);

    // Replay buffered messages for this type
    var remaining = [];
    for (var i = 0; i < buffer.length; i++) {
      if (buffer[i].type === type) {
        try { callback(buffer[i].data); } catch (e) { /* ignore */ }
      } else {
        remaining.push(buffer[i]);
      }
    }
    buffer = remaining;
  };

  /**
   * Send a typed message to the server.
   */
  window.wsSend = function (type, data) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: type, data: data }));
    } else {
      console.warn("[ws] not connected, message dropped");
    }
  };

  document.addEventListener("DOMContentLoaded", connect);
})();
