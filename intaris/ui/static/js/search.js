/**
 * Search tab — three-kind conversation search (summary, intention,
 * reasoning) over canonical Intaris tables. Optional vector tier
 * (pgvector or Qdrant native dense+sparse hybrid) when configured.
 */
function searchTab() {
  return {
    initialized: false,
    loading: false,
    submitted: false,

    // Form state
    q: '',
    mode: 'auto',
    view: 'sessions',
    filterAgentId: '',
    filterSessionId: '',
    filterFromTs: '',
    filterToTs: '',
    filterKinds: [],
    availableKinds: ['summary', 'intention', 'reasoning'],

    // Results
    matches: [],
    sessions: [],
    nextCursor: null,
    lastResponse: null,
    degradedReason: '',

    // Health
    health: null,
    _healthInterval: null,

    init() {
      window.addEventListener('intaris:tab-changed', (e) => {
        if (e.detail.tab === 'search' && !this.initialized) {
          this.initialized = true;
          this.refreshHealth();
          this._healthInterval = setInterval(() => this.refreshHealth(), 30000);
        }
      });
      window.addEventListener('intaris:user-changed', () => {
        if (this.initialized) {
          this.matches = [];
          this.sessions = [];
          this.submitted = false;
          this.refreshHealth();
        }
      });
    },

    async refreshHealth() {
      try {
        this.health = await IntarisAPI.getSearchHealth();
      } catch (e) {
        this.health = {
          enabled: false,
          lexical: { backend: 'disabled' },
          vector: { provider: 'disabled' },
        };
      }
    },

    toggleKind(kind) {
      const idx = this.filterKinds.indexOf(kind);
      if (idx >= 0) {
        this.filterKinds.splice(idx, 1);
      } else {
        this.filterKinds.push(kind);
      }
    },

    _buildBody(cursor = null) {
      const filters = {};
      if (this.filterAgentId.trim()) filters.agent_id = this.filterAgentId.trim();
      if (this.filterSessionId.trim()) filters.session_id = this.filterSessionId.trim();
      if (this.filterFromTs) filters.from_ts = new Date(this.filterFromTs).toISOString();
      if (this.filterToTs) filters.to_ts = new Date(this.filterToTs).toISOString();

      return {
        q: this.q.trim(),
        filters,
        kinds: this.filterKinds.length > 0 ? [...this.filterKinds] : null,
        mode: this.mode,
        limit: this.view === 'sessions' ? 25 : 50,
        cursor,
      };
    },

    async run() {
      if (!this.q.trim()) return;
      this.loading = true;
      this.submitted = true;
      this.matches = [];
      this.sessions = [];
      this.nextCursor = null;
      this.degradedReason = '';
      try {
        const body = this._buildBody(null);
        const result = this.view === 'sessions'
          ? await IntarisAPI.searchSessions(body)
          : await IntarisAPI.searchMatches(body);
        this.lastResponse = result;
        this.nextCursor = result.next_cursor || null;
        if (this.view === 'sessions') {
          this.sessions = result.sessions || [];
        } else {
          this.matches = result.matches || [];
        }
        this._extractDegradedReason(result);
      } catch (e) {
        Alpine.store('notify').error('Search failed: ' + (e.message || e));
      } finally {
        this.loading = false;
      }
    },

    async loadMore() {
      if (!this.nextCursor || this.loading) return;
      this.loading = true;
      try {
        const body = this._buildBody(this.nextCursor);
        const result = this.view === 'sessions'
          ? await IntarisAPI.searchSessions(body)
          : await IntarisAPI.searchMatches(body);
        this.nextCursor = result.next_cursor || null;
        if (this.view === 'sessions') {
          this.sessions.push(...(result.sessions || []));
        } else {
          this.matches.push(...(result.matches || []));
        }
        this._extractDegradedReason(result);
      } catch (e) {
        Alpine.store('notify').error('Load more failed: ' + (e.message || e));
      } finally {
        this.loading = false;
      }
    },

    _extractDegradedReason(result) {
      this.degradedReason = '';
      const used = result?.backend?.mode_used;
      const requested = this.mode;
      if (requested === 'vector' && used === 'lexical') {
        this.degradedReason = 'vector_unavailable';
      } else if (requested === 'hybrid' && used === 'lexical') {
        this.degradedReason = 'vector_unavailable';
      }
    },

    /**
     * Sanitize and render snippet HTML. Allow only <mark> tags through.
     */
    renderSnippet(snippet) {
      if (!snippet) return '';
      const escaped = snippet
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/&lt;mark&gt;/g, '<mark>')
        .replace(/&lt;\/mark&gt;/g, '</mark>');
      return escaped;
    },
  };
}


/**
 * Settings card surfacing search subsystem health and reindex action.
 */
function searchSettings() {
  return {
    searchHealth: null,
    reindexing: false,
    _interval: null,

    init() {
      this.refresh();
      window.addEventListener('intaris:tab-changed', (e) => {
        if (e.detail.tab === 'settings') {
          this.refresh();
          if (!this._interval) {
            this._interval = setInterval(() => this.refresh(), 10000);
          }
        } else if (this._interval) {
          clearInterval(this._interval);
          this._interval = null;
        }
      });
    },

    async refresh() {
      try {
        this.searchHealth = await IntarisAPI.getSearchHealth();
      } catch (e) {
        this.searchHealth = {
          enabled: false,
          lexical: { backend: 'disabled' },
          vector: { provider: 'disabled' },
        };
      }
    },

    async triggerReindex() {
      this.reindexing = true;
      try {
        const result = await IntarisAPI.triggerSearchReindex();
        Alpine.store('notify').success('Reindex queued: ' + result.job_id);
        await this.refresh();
      } catch (e) {
        Alpine.store('notify').error('Reindex failed: ' + (e.message || e));
      } finally {
        this.reindexing = false;
      }
    },
  };
}
