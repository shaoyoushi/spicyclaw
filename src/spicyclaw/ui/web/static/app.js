const { createApp, ref, reactive, computed, watch, nextTick, onMounted, onUnmounted } = Vue;

const app = createApp({
  setup() {
    // --- State ---
    const sessions = ref([]);
    const currentSessionId = ref(null);
    const displayMessages = ref([]);
    const inputText = ref('');
    const status = ref('stopped');
    const wsConnected = ref(false);
    const chatView = ref(null);
    const inputEl = ref(null);

    let ws = null;
    let reconnectTimer = null;
    let autoScroll = true;

    // --- Time formatting ---
    function formatTime(ts) {
      if (!ts) return '';
      const d = new Date(ts * 1000);
      const pad = n => String(n).padStart(2, '0');
      return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }

    function formatDateTime(ts) {
      if (!ts) return '';
      const d = new Date(ts * 1000);
      const pad = n => String(n).padStart(2, '0');
      return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }

    // --- API helpers ---
    async function api(method, path, body) {
      const opts = { method, headers: { 'Content-Type': 'application/json' } };
      if (body !== undefined) opts.body = JSON.stringify(body);
      const resp = await fetch('/api' + path, opts);
      if (!resp.ok) throw new Error(`API ${resp.status}: ${await resp.text()}`);
      return resp.json();
    }

    // --- Session management ---
    async function fetchSessions() {
      sessions.value = await api('GET', '/sessions');
    }

    async function createSession() {
      const s = await api('POST', '/sessions', {});
      sessions.value.push(s);
      await switchSession(s.id);
    }

    async function switchSession(id) {
      if (currentSessionId.value === id) return;
      disconnectWS();
      currentSessionId.value = id;
      await loadMessages(id);
      connectWS(id);
      nextTick(() => scrollToBottom(true));
    }

    async function loadMessages(id) {
      const ctx = await api('GET', `/sessions/${id}/context`);
      displayMessages.value = contextToMessages(ctx);
    }

    // --- Transform raw context to display messages ---
    function contextToMessages(ctx) {
      const msgs = [];
      const toolCallMap = {};  // tool_call_id -> tool call info from assistant msg

      for (const m of ctx) {
        if (m.role === 'system') continue;

        const ts = m.ts || null;

        if (m.role === 'user') {
          msgs.push({ type: 'user', content: m.content || '', ts });
        } else if (m.role === 'assistant') {
          if (m.content) {
            msgs.push({ type: 'assistant', content: m.content, streaming: false, ts });
          }
          if (m.tool_calls) {
            for (const tc of m.tool_calls) {
              let args = {};
              try { args = JSON.parse(tc.arguments); } catch {}
              const workNode = args.work_node || '';
              const nextStep = args.next_step || '';
              delete args.work_node;
              delete args.next_step;

              const command = args.command || '';
              delete args.command;
              const reason = args.reason || '';
              delete args.reason;

              const argsDisplay = Object.keys(args).length > 0 ? JSON.stringify(args, null, 2) : '';

              toolCallMap[tc.id] = msgs.length;
              msgs.push({
                type: 'tool',
                name: tc.function_name,
                toolCallId: tc.id,
                command: command,
                argsDisplay: tc.function_name === 'stop' ? reason : argsDisplay,
                workNode,
                nextStep,
                output: null,
                error: null,
                returnCode: null,
                loading: true,
                collapsed: false,
                ts,
              });
            }
          }
        } else if (m.role === 'tool') {
          const idx = toolCallMap[m.tool_call_id];
          if (idx !== undefined && msgs[idx]) {
            const toolMsg = msgs[idx];
            // Parse content: output + optional [STDERR] section
            const content = m.content || '';
            const stderrIdx = content.indexOf('\n[STDERR]\n');
            if (stderrIdx >= 0) {
              toolMsg.output = content.slice(0, stderrIdx);
              toolMsg.error = content.slice(stderrIdx + 10);
            } else {
              toolMsg.output = content;
            }
            toolMsg.returnCode = 0;  // approximate, exact rc comes from events
            toolMsg.loading = false;
            toolMsg.collapsed = toolMsg.output.length > 300;
          }
        }
      }
      return msgs;
    }

    // --- WebSocket ---
    function connectWS(id) {
      if (!id) return;
      disconnectWS();
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${proto}//${location.host}/api/sessions/${id}/ws`;
      ws = new WebSocket(url);

      ws.onopen = () => {
        wsConnected.value = true;
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      };

      ws.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data);
          handleServerEvent(event);
        } catch {}
      };

      ws.onclose = () => {
        wsConnected.value = false;
        ws = null;
        scheduleReconnect(id);
      };

      ws.onerror = () => { /* onclose will fire */ };
    }

    function disconnectWS() {
      if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
      if (ws) { ws.onclose = null; ws.close(); ws = null; }
      wsConnected.value = false;
    }

    function scheduleReconnect(id) {
      if (reconnectTimer) return;
      reconnectTimer = setTimeout(async () => {
        reconnectTimer = null;
        if (currentSessionId.value === id) {
          await loadMessages(id);
          connectWS(id);
        }
      }, 2000);
    }

    function wsSend(type, data) {
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      ws.send(JSON.stringify({
        type,
        session_id: currentSessionId.value,
        data: data || {},
      }));
    }

    // --- Handle server events ---
    function handleServerEvent(event) {
      const msgs = displayMessages.value;
      const eventTs = event.ts || (Date.now() / 1000);

      switch (event.type) {
        case 'chunk': {
          const text = event.data.text || '';
          const last = msgs[msgs.length - 1];
          if (last && last.type === 'assistant' && last.streaming) {
            last.content += text;
          } else {
            msgs.push({ type: 'assistant', content: text, streaming: true, ts: eventTs });
          }
          maybeScroll();
          break;
        }

        case 'tool_call': {
          // Finalize any streaming assistant message
          finalizeStreaming();
          const d = event.data;
          msgs.push({
            type: 'tool',
            name: d.name,
            toolCallId: d.tool_call_id,
            command: (d.arguments || {}).command || '',
            argsDisplay: d.name === 'stop'
              ? (d.arguments || {}).reason || ''
              : formatArgs(d.arguments),
            workNode: d.work_node || '',
            nextStep: d.next_step || '',
            output: null,
            error: null,
            returnCode: null,
            loading: true,
            collapsed: false,
            ts: eventTs,
          });
          maybeScroll();
          break;
        }

        case 'tool_end': {
          const tcId = event.data.tool_call_id;
          const toolMsg = [...msgs].reverse().find(m => m.type === 'tool' && m.toolCallId === tcId);
          if (toolMsg) {
            toolMsg.output = event.data.output || '';
            toolMsg.error = event.data.error || '';
            toolMsg.returnCode = event.data.return_code ?? 0;
            toolMsg.loading = false;
            toolMsg.collapsed = toolMsg.output.length > 300;
          }
          maybeScroll();
          break;
        }

        case 'status': {
          status.value = event.data.status || 'stopped';
          if (status.value === 'stopped') {
            finalizeStreaming();
          }
          break;
        }

        case 'session_update': {
          const sid = event.session_id;
          const s = sessions.value.find(s => s.id === sid);
          if (s && event.data.title) {
            s.title = event.data.title;
          }
          break;
        }

        case 'system': {
          msgs.push({ type: 'system', content: event.data.message || '', ts: eventTs });
          maybeScroll();
          break;
        }

        case 'error': {
          msgs.push({ type: 'error', content: event.data.message || 'Unknown error', ts: eventTs });
          maybeScroll();
          break;
        }
      }
    }

    function finalizeStreaming() {
      const msgs = displayMessages.value;
      const last = msgs[msgs.length - 1];
      if (last && last.type === 'assistant' && last.streaming) {
        last.streaming = false;
      }
    }

    function formatArgs(args) {
      if (!args || Object.keys(args).length === 0) return '';
      const filtered = { ...args };
      delete filtered.command;
      delete filtered.reason;
      return Object.keys(filtered).length > 0 ? JSON.stringify(filtered, null, 2) : '';
    }

    // --- User actions ---
    function sendMessage() {
      const text = inputText.value.trim();
      if (!text) return;

      // Handle /commands — always allowed
      if (text.startsWith('/')) {
        const parts = text.slice(1).split(/\s+/, 2);
        const cmd = parts[0];
        const args = text.slice(1 + cmd.length).trim();
        wsSend('command', { command: cmd, args: args });
        displayMessages.value.push({ type: 'system', content: `> /${text.slice(1)}`, ts: Date.now() / 1000 });
        inputText.value = '';
        nextTick(() => adjustHeight());
        return;
      }

      if (status.value === 'paused') {
        // In step mode, confirm execution
        wsSend('confirm', {});
        displayMessages.value.push({ type: 'system', content: '> Confirmed', ts: Date.now() / 1000 });
        inputText.value = '';
        nextTick(() => adjustHeight());
        return;
      }

      // Allow sending messages even when running (they get queued)
      const nowTs = Date.now() / 1000;
      displayMessages.value.push({ type: 'user', content: text, ts: nowTs });
      wsSend('message', { content: text });
      inputText.value = '';
      nextTick(() => {
        adjustHeight();
        scrollToBottom(true);
      });
    }

    function abort() {
      wsSend('abort', {});
    }

    function confirm() {
      wsSend('confirm', {});
    }

    function handleKeydown(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    }

    function adjustHeight() {
      const el = inputEl.value;
      if (!el) return;
      el.style.height = 'auto';
      el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    }

    // --- Scroll ---
    function onScroll() {
      const el = chatView.value;
      if (!el) return;
      autoScroll = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    }

    function maybeScroll() {
      if (autoScroll) scrollToBottom(false);
    }

    function scrollToBottom(force) {
      nextTick(() => {
        const el = chatView.value;
        if (!el) return;
        if (force || autoScroll) {
          el.scrollTop = el.scrollHeight;
        }
      });
    }

    // --- Computed ---
    const isRunning = computed(() => status.value === 'thinking' || status.value === 'executing');
    const isPaused = computed(() => status.value === 'paused');
    const canSend = computed(() => true);  // Always allow sending

    // --- Lifecycle ---
    onMounted(async () => {
      await fetchSessions();
      if (sessions.value.length > 0) {
        await switchSession(sessions.value[0].id);
      }
    });

    onUnmounted(() => {
      disconnectWS();
    });

    return {
      sessions,
      currentSessionId,
      displayMessages,
      inputText,
      status,
      wsConnected,
      chatView,
      inputEl,
      isRunning,
      isPaused,
      canSend,
      createSession,
      switchSession,
      sendMessage,
      abort,
      confirm,
      handleKeydown,
      adjustHeight,
      onScroll,
      formatTime,
      formatDateTime,
    };
  },
});

app.mount('#app');
