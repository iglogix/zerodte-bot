/* Chat UI client for the 0DTE bot dashboard */

(function () {
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");

  let _pendingConfirm = null;

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function addMessage(role, content, extra) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("chat-msg", `chat-msg-${role}`);

    if (role === "user") {
      wrapper.innerHTML = `<strong>You:</strong> ${escapeHtml(content)}`;
    } else if (role === "assistant") {
      wrapper.innerHTML = `<strong>Bot:</strong> ${formatContent(content)}`;
    } else if (role === "tool") {
      wrapper.innerHTML = `<em>Tool result:</em> <pre>${escapeHtml(content)}</pre>`;
    } else if (role === "confirm") {
      wrapper.innerHTML =
        `<strong>Bot:</strong> ${escapeHtml(content)}` +
        `<div class="confirm-buttons" id="confirm-ui">` +
        `<button class="btn btn-sm btn-success me-2" onclick="window.confirmAction(true)">Yes</button>` +
        `<button class="btn btn-sm btn-secondary" onclick="window.confirmAction(false)">Cancel</button>` +
        `</div>`;
    }

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function formatContent(text) {
    if (!text) return "";
    // Basic markdown-ish: bold **text** and code `text`
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`(.*?)`/g, "<code>$1</code>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  // WebSocket message handler for chat responses
  if (typeof wsOn === "function") {
    wsOn("chat_response", function (data) {
      if (data.type === "text") {
        addMessage("assistant", data.content);
      } else if (data.type === "tool_call") {
        addMessage("tool", JSON.stringify(data.result, null, 2));
      } else if (data.type === "confirmation") {
        _pendingConfirm = {
          tool: data.tool,
          args: data.args,
        };
        addMessage("confirm", data.message);
      }
    });
  }

  // Send chat message
  window.sendChat = function (e) {
    if (e) e.preventDefault();

    const text = chatInput.value.trim();
    if (!text) return;

    addMessage("user", text);
    chatInput.value = "";

    if (typeof wsSend === "function") {
      wsSend("chat_message", { message: text });
    }
  };

  // Confirm or cancel an action
  window.confirmAction = function (yes) {
    // Remove confirm buttons
    const confirmUi = document.getElementById("confirm-ui");
    if (confirmUi) {
      confirmUi.remove();
    }

    if (yes && _pendingConfirm) {
      addMessage("user", "Confirmed.");
      if (typeof wsSend === "function") {
        wsSend("chat_confirm", {
          tool: _pendingConfirm.tool,
          args: _pendingConfirm.args,
          confirmed: true,
        });
      }
    } else {
      addMessage("user", "Cancelled.");
    }

    _pendingConfirm = null;
  };

  // Allow Enter to send (Shift+Enter for newline)
  if (chatInput) {
    chatInput.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        window.sendChat();
      }
    });
  }
})();
