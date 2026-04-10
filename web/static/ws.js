// WebSocket client — multiplexed message hub with auto-reconnect.
(function () {
  "use strict";

  var handlers = {};
  var socket = null;
  var reconnectTimer = null;

  function getWsUrl() {
    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    return proto + "//" + location.host + "/ws";
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
      var type = msg.type;
      if (type && handlers[type]) {
        for (var i = 0; i < handlers[type].length; i++) {
          try {
            handlers[type][i](msg.data);
          } catch (e) {
            console.error("[ws] handler error for type '" + type + "':", e);
          }
        }
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
   * @param {string} type - Message type to listen for.
   * @param {function} callback - Called with the message data payload.
   */
  window.wsOn = function (type, callback) {
    if (!handlers[type]) {
      handlers[type] = [];
    }
    handlers[type].push(callback);
  };

  /**
   * Send a typed message to the server.
   * @param {string} type - Message type.
   * @param {*} data - Payload data.
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
