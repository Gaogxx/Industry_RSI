/* Kaizen 控制台 v0.5 —— 参考 Claude / Codex：主区域默认是对话（AG-UI 1.0）；案例、CAPA、自我进化等功能放在左侧栏，点开后在同一主区域完整显示（零依赖，内网可用） */
(function () {
  "use strict";

  var META = null, LAST_CLOSE = {};
  var view = document.getElementById("view");

  // ── 工具 ───────────────────────────────────────────────
  function esc(s) { return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]; }); }
  function pct(v) { return v == null ? "-" : Math.round(v * 100) + "%"; }
  function d10(iso) { return iso ? new Date(iso).toLocaleDateString("zh-CN") : "-"; }
  function dt(iso) { return iso ? new Date(iso).toLocaleString("zh-CN", { hour12: false }).slice(0, 16) : "-"; }
  function code(id) { return "KZ-" + String(id).padStart(4, "0"); }
  function toast(msg) { var t = document.getElementById("toast"); t.textContent = msg; t.classList.add("show"); clearTimeout(t._h); t._h = setTimeout(function () { t.classList.remove("show"); }, 2800); }
  function lbl(group, key) { return (META && META[group] && META[group][key]) || key || "-"; }
  function symName(s) { return (META.symptoms[s] || {}).name || s || "-"; }
  function sevBadge(s) { return '<span class="badge ' + esc(s) + '">' + esc(s) + " " + esc(lbl("severities", s)) + "</span>"; }
  function statusBadge(s) { var c = { closed: "ok", open: "warn" }[s] || "accent"; return '<span class="badge ' + c + '">' + esc(lbl("case_statuses", s)) + "</span>"; }
  function key() { try { return localStorage.getItem("kaizen_key") || ""; } catch (e) { return ""; } }
  function setKey(k) { try { localStorage.setItem("kaizen_key", k); } catch (e) {} }

  function api(path, opts, retried) {
    opts = opts || {};
    var headers = Object.assign({ "Content-Type": "application/json" }, opts.headers || {});
    if (key()) headers.Authorization = "Bearer " + key();
    return fetch("api/" + path, Object.assign({}, opts, { headers: headers })).then(function (r) {
      if (r.status === 401 && !retried) { var k = window.prompt("此操作需要 Kaizen API Key（KAIZEN_API_KEY）："); if (k) { setKey(k.trim()); return api(path, opts, true); } }
      if (!r.ok) return r.json().catch(function () { return {}; }).then(function (b) { throw new Error(b.detail || r.statusText); });
      return opts.raw ? r : r.json();
    });
  }
  function post(path, body) { return api(path, { method: "POST", body: JSON.stringify(body || {}) }); }

  function md(src) {
    var lines = String(src || "").split("\n"), out = [], i = 0;
    function inline(s) {
      return esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, t, u) { return /^(https?:|#|\/)/.test(u) ? '<a href="' + u + '" target="_blank" rel="noopener">' + t + "</a>" : t; });
    }
    while (i < lines.length) {
      var l = lines[i], m;
      if (/^\|/.test(l)) {
        var rows = []; while (i < lines.length && /^\|/.test(lines[i])) { rows.push(lines[i]); i++; }
        var cells = function (r) { return r.replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); }); };
        var h = '<div class="table-wrap"><table><thead><tr>' + cells(rows[0]).map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") + "</tr></thead><tbody>";
        rows.slice(2).forEach(function (r) { h += "<tr>" + cells(r).map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>"; });
        out.push(h + "</tbody></table></div>"); continue;
      }
      if ((m = /^(#{1,4})\s+(.*)/.exec(l))) out.push("<h" + m[1].length + ">" + inline(m[2]) + "</h" + m[1].length + ">");
      else if (/^>\s?/.test(l)) { var q = []; while (i < lines.length && /^>\s?/.test(lines[i])) { q.push(inline(lines[i].replace(/^>\s?/, ""))); i++; } out.push("<blockquote>" + q.join("<br>") + "</blockquote>"); continue; }
      else if (/^\s*([-*]|\d+\.)\s+/.test(l)) { var ord = /^\s*\d+\./.test(l), it = []; while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) { it.push("<li>" + inline(lines[i].replace(/^\s*([-*]|\d+\.)\s+/, "")) + "</li>"); i++; } out.push((ord ? "<ol>" : "<ul>") + it.join("") + (ord ? "</ol>" : "</ul>")); continue; }
      else if (/^---+$/.test(l.trim())) out.push("<hr>");
      else if (l.trim()) out.push("<p>" + inline(l) + "</p>");
      i++;
    }
    return '<div class="md">' + out.join("\n") + "</div>";
  }

  // ── 图表（纯 SVG） ─────────────────────────────────────
  function learningChart(points, opts) {
    opts = opts || {};
    if (!points || !points.length) return '<div class="empty">暂无已验证案例</div>';
    var W = 680, H = opts.h || 250, pl = 36, pr = 64, pt = 14, pb = 28, n = points.length;
    var x = function (i) { return pl + (i - 1) / Math.max(1, n - 1) * (W - pl - pr); };
    var y = function (v) { return pt + (1 - v) * (H - pt - pb); };
    var line = function (k) { return points.map(function (p) { return x(p.i).toFixed(1) + "," + y(p[k]).toFixed(1); }).join(" "); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="学习曲线：持续学习与冻结先验的首位根因命中率">';
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) { s += '<line class="axis" x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y(g) + '" y2="' + y(g) + '"/><text x="' + (pl - 6) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + g * 100 + "%</text>"; });
    if (opts.window !== false) {
      s += '<polyline fill="none" stroke="var(--chart-1)" stroke-width="1" opacity=".35" points="' + line("win_learning") + '"/>';
      s += '<polyline fill="none" stroke="var(--muted)" stroke-width="1" opacity=".35" stroke-dasharray="3 3" points="' + line("win_frozen") + '"/>';
    }
    s += '<polyline fill="none" stroke="var(--muted)" stroke-width="2" stroke-dasharray="6 4" points="' + line("cum_frozen") + '"/>';
    s += '<polyline fill="none" stroke="var(--chart-1)" stroke-width="2.6" points="' + line("cum_learning") + '"/>';
    var last = points[n - 1];
    s += '<text x="' + (x(n) + 6) + '" y="' + (y(last.cum_learning) + 4) + '" style="fill:var(--chart-1);font-weight:700;font-size:12px">' + pct(last.cum_learning) + "</text>";
    s += '<text x="' + (x(n) + 6) + '" y="' + (y(last.cum_frozen) + 4) + '" style="fill:var(--muted);font-size:12px">' + pct(last.cum_frozen) + "</text>";
    var step = Math.max(1, Math.round(n / 24));
    points.forEach(function (p) { if (p.i % step === 0 || p.i === n) s += '<circle cx="' + x(p.i) + '" cy="' + y(p.cum_learning) + '" r="6" fill="transparent"><title>第 ' + p.i + " 个案例 " + esc(p.case) + "（" + esc(symName(p.symptom)) + "）\n累计命中率：持续学习 " + pct(p.cum_learning) + " · 冻结先验 " + pct(p.cum_frozen) + "\n本案：持续学习" + (p.hit_learning ? "✔" : "✘") + " 冻结" + (p.hit_frozen ? "✔" : "✘") + "</title></circle>"; });
    [1, Math.round(n / 2), n].forEach(function (i) { s += '<text x="' + x(i) + '" y="' + (H - 8) + '" text-anchor="middle">第 ' + i + " 案</text>"; });
    return s + '</svg><div class="legend"><span><i style="background:var(--chart-1)"></i>持续学习（累计）</span><span><i style="background:var(--muted)"></i>冻结在教科书先验（累计，虚线）</span>' +
      (opts.window !== false ? '<span><i style="background:var(--chart-1);opacity:.35"></i>最近 10 案滑动窗口</span>' : "") + "</div>";
  }

  function trendChart(trend) {
    var W = 560, H = 190, pad = 26, max = Math.max.apply(null, trend.map(function (t) { return Math.max(t.opened, t.closed); }).concat([1])), bw = (W - pad * 2) / trend.length;
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="月度新增与关闭案例"><line class="axis" x1="' + pad + '" y1="' + (H - 22) + '" x2="' + (W - 6) + '" y2="' + (H - 22) + '"/>';
    trend.forEach(function (t, i) {
      var x = pad + i * bw + bw * 0.15, w = bw * 0.33, sc = (H - 46) / max;
      s += '<rect x="' + x + '" y="' + (H - 22 - t.opened * sc) + '" width="' + w + '" height="' + t.opened * sc + '" rx="2" fill="var(--chart-1)"><title>' + t.month + " 新增 " + t.opened + "</title></rect>";
      s += '<rect x="' + (x + w + 1) + '" y="' + (H - 22 - t.closed * sc) + '" width="' + w + '" height="' + t.closed * sc + '" rx="2" fill="var(--chart-2)"><title>' + t.month + " 关闭 " + t.closed + "</title></rect>";
      s += '<text x="' + (x + w) + '" y="' + (H - 7) + '" text-anchor="middle">' + t.month.slice(5) + "</text>";
    });
    return s + '</svg><div class="legend"><span><i style="background:var(--chart-1)"></i>新增</span><span><i style="background:var(--chart-2)"></i>关闭</span></div>';
  }

  function hbars(items, fmt) {
    if (!items.length) return '<div class="empty">暂无数据</div>';
    var max = Math.max.apply(null, items.map(function (x) { return x[1]; }));
    return items.map(function (x) { return '<div class="hbar"><span class="lbl" title="' + esc(x[0]) + '">' + esc(x[0]) + '</span><span class="track"><span class="fill" style="display:block;width:' + Math.max(3, x[1] / max * 100) + '%"></span></span><span class="num muted">' + (fmt ? fmt(x) : x[1]) + "</span></div>"; }).join("");
  }

  function waferSVG(rows) {
    var n = rows.length, c = 6, s = '<svg viewBox="0 0 ' + n * c + " " + n * c + '" role="img" aria-label="晶圆图">';
    s += '<circle class="wafer-edge" cx="' + n * c / 2 + '" cy="' + n * c / 2 + '" r="' + (n * c / 2 - 0.5) + '"/>';
    rows.forEach(function (row, i) { for (var j = 0; j < row.length; j++) if (row[j] !== ".") s += '<rect x="' + j * c + '" y="' + i * c + '" width="' + (c - 0.8) + '" height="' + (c - 0.8) + '" class="' + (row[j] === "1" ? "die-fail" : "die-pass") + '"/>'; });
    return s + "</svg>";
  }

  function seriesChart(ds) {
    var pts = ds.series || []; if (!pts.length) return "";
    var W = 680, H = 200, pl = 40, pr = 10, pt = 10, pb = 24;
    var ts = pts.map(function (p) { return Date.parse(p.time); }), t0 = Math.min.apply(null, ts), t1 = Math.max.apply(null, ts);
    var vs = pts.map(function (p) { return p.v; }), v0 = Math.min.apply(null, vs), v1 = Math.max.apply(null, vs);
    var x = function (t) { return pl + (t - t0) / Math.max(1, t1 - t0) * (W - pl - pr); }, y = function (v) { return pt + (1 - (v - v0) / Math.max(1e-9, v1 - v0)) * (H - pt - pb); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="批次指标时间序列">';
    [v0, (v0 + v1) / 2, v1].forEach(function (g) { s += '<line class="axis" x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y(g) + '" y2="' + y(g) + '"/><text x="' + (pl - 4) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + g.toFixed(1) + "</text>"; });
    (ds.events || []).forEach(function (e) { var ex = x(Date.parse(e.time)); if (ex >= pl && ex <= W - pr) s += '<line x1="' + ex + '" x2="' + ex + '" y1="' + pt + '" y2="' + (H - pb) + '" stroke="var(--chart-line)" stroke-dasharray="3 3" opacity=".7"/><text x="' + (ex + 3) + '" y="' + (pt + 10) + '" style="fill:var(--chart-line)">PM ' + esc(e.tool) + "</text>"; });
    pts.forEach(function (p) { s += '<circle cx="' + x(Date.parse(p.time)).toFixed(1) + '" cy="' + y(p.v).toFixed(1) + '" r="' + (p.hit ? 3.2 : 2.2) + '" fill="' + (p.fail ? "var(--warn)" : "var(--chart-2)") + '"' + (p.hit ? ' stroke="var(--chart-1)" stroke-width="1.4"' : "") + "/>"; });
    s += '<text x="' + pl + '" y="' + (H - 6) + '">' + d10(new Date(t0).toISOString()) + '</text><text x="' + (W - pr) + '" y="' + (H - 6) + '" text-anchor="end">' + d10(new Date(t1).toISOString()) + "</text>";
    return s + '</svg><div class="legend"><span><i style="background:var(--warn)"></i>失效批次</span><span><i style="background:var(--chart-2)"></i>正常批次</span><span><i style="border:1.5px solid var(--chart-1);background:transparent"></i>经过可疑机台 ' + esc(ds.fdc_tool || "") + '</span><span><i style="background:var(--chart-line)"></i>PM 事件</span></div>';
  }

  function fdcChart(rows) {
    if (!rows || rows.length < 4) return "";
    var params = Object.keys(rows[0]).filter(function (k) { return k !== "lot" && k !== "time"; });
    return params.slice(0, 2).map(function (p) {
      var W = 330, H = 120, pad = 8, vs = rows.map(function (r) { return r[p]; }), v0 = Math.min.apply(null, vs), v1 = Math.max.apply(null, vs);
      var pts = rows.map(function (r, i) { return (pad + i / (rows.length - 1) * (W - 2 * pad)).toFixed(1) + "," + (pad + (1 - (r[p] - v0) / Math.max(1e-9, v1 - v0)) * (H - 2 * pad - 14)).toFixed(1); }).join(" ");
      return '<div><svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="FDC ' + esc(p) + '"><polyline fill="none" stroke="var(--chart-1)" stroke-width="1.6" points="' + pts + '"/><text x="' + pad + '" y="' + (H - 3) + '">' + esc(p) + "（" + v0.toFixed(2) + " ~ " + v1.toFixed(2) + "）</text></svg></div>";
    }).join("");
  }

  function hypList(hyps, cf) {
    return hyps.map(function (h, i) {
      var chips = h.factors.map(function (f) {
        var cls = f.name === "先验知识" ? "" : f.x > 1 ? "up" : f.x < 1 ? "down" : "";
        return '<span class="chip ' + cls + '" title="' + esc(f.detail) + '">' + esc(f.name) + (f.name === "先验知识" ? " " + pct(f.x) : " ×" + f.x) + "</span>";
      }).join("");
      var frozen = cf && cf.ranking_frozen ? cf.ranking_frozen.find(function (x) { return x.cause_code === h.cause_code; }) : null;
      return '<div class="hyp' + (i === 0 ? " lead" : "") + '"><div class="row"><span><b>' + (i + 1) + ". " + esc(h.name) + '</b> <span class="badge">' + esc(lbl("six_m", h.six_m)) + "</span>" +
        (h.learned ? ' <span class="badge accent">新学到</span>' : "") + '</span><span class="num" style="text-align:right"><b>' + pct(h.posterior) + "</b></span></div>" +
        '<div class="bar"><i style="width:' + Math.max(2, h.posterior * 100) + '%"></i></div><div class="chips">' + chips +
        (frozen ? '<span class="chip">若不学习 ' + pct(frozen.posterior) + "</span>" : "") + "</div></div>";
    }).join("");
  }

  function fishbone(ish, rootCats, problem) {
    var W = 940, H = 380, mid = 190, cats = [["man", "machine", "material"], ["method", "environment", "measurement"]];
    var s = '<svg class="fishbone" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="鱼骨图"><line class="spine" x1="30" y1="' + mid + '" x2="742" y2="' + mid + '"/><rect class="head" x="742" y="' + (mid - 46) + '" width="190" height="92" rx="10"/>';
    var words = String(problem || ""); for (var k = 0; k < Math.min(4, Math.ceil(words.length / 13)); k++) s += '<text x="837" y="' + (mid - 22 + k * 18) + '" text-anchor="middle" style="fill:var(--fg);font-size:12px">' + esc(words.slice(k * 13, k * 13 + 13)) + "</text>";
    cats.forEach(function (row, r) { row.forEach(function (c, j) {
      var x0 = 175 + j * 205, y0 = r === 0 ? 34 : H - 34, x1 = x0 + 110, y1 = r === 0 ? mid - 4 : mid + 4, hit = rootCats.indexOf(c) >= 0;
      s += '<line class="bone" x1="' + x0 + '" y1="' + y0 + '" x2="' + x1 + '" y2="' + y1 + '"' + (hit ? ' style="stroke:var(--warn);stroke-width:2.5"' : "") + "/>";
      s += '<text class="cat" x="' + x0 + '" y="' + (r === 0 ? y0 - 10 : y0 + 20) + '" text-anchor="middle"' + (hit ? ' style="fill:var(--warn)"' : "") + ">" + esc(lbl("six_m", c)) + " · " + c + (hit ? " ★" : "") + "</text>";
      (ish[c] || []).slice(0, 4).forEach(function (cause, kk) { var t = 0.2 + kk * 0.2, px = x0 + (x1 - x0) * t, py = y0 + (y1 - y0) * t, txt = cause.replace(/（\d+%）$/, ""); s += '<text class="cause" x="' + (px - 10) + '" y="' + (py + 4) + '" text-anchor="end"><title>' + esc(cause) + "</title>" + esc(txt.length > 12 ? txt.slice(0, 12) + "…" : txt) + "</text>"; });
    }); });
    return s + "</svg>";
  }

  function caseTable(cases, compact) {
    if (!cases.length) return '<div class="empty">暂无案例</div>';
    return '<div class="table-wrap"><table><thead><tr><th>编号</th><th>标题</th>' + (compact ? "" : "<th>工段</th><th>症状</th>") + "<th>严重度</th><th>状态</th><th>根因</th>" + (compact ? "" : "<th>创建</th>") + "</tr></thead><tbody>" +
      cases.map(function (c) {
        return '<tr class="click" data-href="#/cases/' + c.id + '"><td class="num">' + esc(c.code) + "</td><td>" + esc(c.title) + "</td>" +
          (compact ? "" : "<td>" + esc(lbl("stages", c.stage)) + "</td><td>" + esc(symName(c.symptom)) + "</td>") + "<td>" + sevBadge(c.severity) + "</td><td>" + statusBadge(c.status) + "</td>" +
          '<td class="muted">' + esc((c.root_cause || "-").slice(0, 18)) + (c.cause_code ? " ✔" : "") + "</td>" + (compact ? "" : '<td class="muted">' + d10(c.created_at) + "</td>") + "</tr>";
      }).join("") + "</tbody></table></div>";
  }
  function bindRows() { view.querySelectorAll("tr[data-href]").forEach(function (tr) { tr.onclick = function () { location.hash = tr.dataset.href; }; }); }

  function eventLine(e) {
    var kind = { confirm: "确认", false_lead: "误导", drift: "产线变更", refute: "否定", edge_added: "新知识", hotspot: "机台热点", memory_add: "新记忆",
      memory_pass: "记忆强化", memory_fail: "失败教训", memory_retire: "记忆停用", memory_manual: "人工" }[e.kind] || e.kind;
    var share = ["confirm", "false_lead", "refute", "edge_added"].indexOf(e.kind) >= 0 ? " <b>" + pct(e.before) + " → " + pct(e.after) + "</b>" : "";
    var src = e.source === "sandbox" ? '<span class="badge accent">孪生</span> ' : e.source === "real" ? '<span class="badge ok">产线</span> ' : "";
    return '<li class="' + esc(e.kind) + '">' + src + '<span class="badge">' + kind + "</span> " + (e.case_code ? '<a href="#/cases/' + e.case_id + '">' + esc(e.case_code) + "</a> " : "") +
      (e.rsi_task_id ? '<span class="muted">练习题 #' + e.rsi_task_id + "</span> " : "") + esc(e.detail) + share +
      (e.symptom_name ? ' <span class="muted">· ' + esc(e.symptom_name) + "</span>" : "") + ' <span class="muted">' + dt(e.created_at) + "</span></li>";
  }

  // ── 总览 ───────────────────────────────────────────────
  function pageOverview() {
    return Promise.all([api("kpis"), api("cases?limit=10"), api("learning/curve")]).then(function (r) {
      var d = r[0], k = d.kpis, L = d.learning, bt = L.backtest || {}, curve = r[2].latest;
      var open = r[1].filter(function (c) { return c.status !== "closed"; });
      var tiles = [[k.cases_total, "案例总数"], [k.cases_open, "进行中"], [k.mttr_days + " 天", "平均关闭周期"], [pct(k.capa_effectiveness), "CAPA 有效率"],
        [k.capa_overdue, "逾期 CAPA", k.capa_overdue > 0], [pct(k.recurrence_rate), "同根因同机台复发率"], [L.events_total, "累计学习事件"], [k.cost_impact_wan >= 10000 ? (k.cost_impact_wan / 10000).toFixed(2) + " 亿" : k.cost_impact_wan + " 万", "累计质量损失（元）"]];
      var cyc = d.latest_cycle;
      view.innerHTML = '<div class="page-head"><h1>质量总览</h1><span class="muted">前道晶圆 → 2.5D 封装 + HBM → FT/BI/SLT → 液冷整机柜</span><span class="spacer"></span><button class="btn" id="run-cycle">▶ 运行改进周期</button></div>' +
        '<div class="tiles">' + tiles.map(function (t) { return '<div class="tile' + (t[2] ? " warn" : "") + '"><div class="v">' + esc(t[0]) + '</div><div class="l">' + t[1] + "</div></div>"; }).join("") + "</div>" +
        '<div class="grid g-side"><div class="panel"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">学习曲线：首位根因命中率</h2><span class="spacer"></span><a href="#/evolution?tab=continual">详情 →</a></div>' +
        '<div class="big-compare"><span class="n l">' + pct(bt.hit1_learning) + '</span><span>持续学习</span><span class="n f">' + pct(bt.hit1_frozen) + '</span><span>冻结在教科书先验</span><span class="muted">' + (bt.cases || 0) + " 个已验证案例 · 预序评估（先预测、后学习）</span></div>" +
        learningChart(curve ? curve.points : [], { h: 220 }) + "</div>" +
        '<div class="panel"><h2>知识更新最大的 5 条</h2>' + (L.top_shifts.map(function (s) { return '<div class="prior-post"><span title="' + esc(s.symptom_name) + '">' + esc(s.name.slice(0, 14)) + '<br><span class="muted" style="font-size:11px">' + esc(s.symptom_name) + '</span></span><span class="pp-track"><span class="prior" style="width:' + s.prior * 100 + '%"></span><span class="post" style="width:' + s.posterior * 100 + '%"></span></span><span class="num">' + pct(s.prior) + " → <b>" + pct(s.posterior) + "</b></span></div>"; }).join("") || '<div class="empty">暂无</div>') +
        '<p class="muted" style="font-size:12px">浅色 = 教科书先验，深色 = 从本厂已验证案例学到的份额</p></div></div>' +
        '<div class="grid g3" style="margin-top:16px"><div class="panel"><h2>已确认根因 Top</h2>' + hbars(d.causes.map(function (c) { return [c.name, c.count, c.cost_wan]; }), function (x) { return x[1] + " 起 · " + x[2] + "万"; }) + "</div>" +
        '<div class="panel"><h2>机台热点（同根因反复）</h2>' + (d.hotspots.map(function (h) { return '<div class="card"><b>' + esc(h.tool) + "</b> × " + h.count + '<div class="meta">' + esc(h.cause) + '</div><div class="meta">' + h.cases.map(function (c) { return esc(c); }).join(" · ") + "</div></div>"; }).join("") || '<div class="empty">暂无</div>') + "</div>" +
        '<div class="panel"><h2>月度案例趋势</h2>' + trendChart(k.trend.slice(-8)) + "</div></div>" +
        '<div class="grid g2" style="margin-top:16px"><div class="panel"><h2>最近一次改进周期' + (cyc ? ' <span class="muted" style="font-weight:400">' + dt(cyc.created_at) + "</span>" : "") + "</h2>" + (cyc ? "<ol>" + cyc.findings.map(function (f) { return "<li>" + esc(f) + "</li>"; }).join("") + "</ol>" : '<div class="empty">尚未运行</div>') + "</div>" +
        '<div class="panel"><h2>进行中的案例</h2>' + caseTable(open, true) + "</div></div>";
      document.getElementById("run-cycle").onclick = function (e) { e.target.disabled = true; e.target.textContent = "运行中…"; post("improvement/run").then(function () { toast("改进周期完成"); route(); }).catch(function (err) { toast(err.message); e.target.disabled = false; }); };
      bindRows();
    });
  }

  // ── 案例列表 ───────────────────────────────────────────
  function pageCases(params) {
    var q = new URLSearchParams(params || "");
    return api("cases?" + q.toString()).then(function (cases) {
      var opt = function (obj, sel, nameFn) { return Object.keys(obj).map(function (k) { return '<option value="' + k + '"' + (sel === k ? " selected" : "") + ">" + esc(nameFn ? nameFn(k) : obj[k]) + "</option>"; }).join(""); };
      view.innerHTML = '<div class="page-head"><h1>案例库</h1><span class="badge">' + cases.length + ' 个</span><span class="spacer"></span>' +
        '<select id="f-st"><option value="">全部工段</option>' + opt(META.stages, q.get("stage")) + "</select>" +
        '<select id="f-sy"><option value="">全部症状</option>' + opt(META.symptoms, q.get("symptom"), symName) + "</select>" +
        '<select id="f-ss"><option value="">全部状态</option>' + opt(META.case_statuses, q.get("status")) + "</select>" +
        '<input id="f-q" placeholder="搜索标题/机台/根因" value="' + esc(q.get("q") || "") + '"><button class="btn" id="new-btn">＋ 新建案例</button></div>' +
        '<div class="panel" id="new-form" hidden><h2>新建案例</h2><form id="cf"><div class="form-grid">' +
        '<label class="f">标题*<input name="title" required minlength="2"></label><label class="f">症状<select name="symptom"><option value="">自动识别</option>' + opt(META.symptoms, null, symName) + "</select></label>" +
        '<label class="f">严重度<select name="severity">' + opt(META.severities, "S2") + '</select></label><label class="f">产品<input name="product" placeholder="AX-900 die"></label>' +
        '<label class="f">工厂<input name="site"></label><label class="f">机台/设备<input name="equipment"></label><label class="f">损失（万元）<input name="cost_impact" type="number" step="0.1" value="0"></label></div>' +
        '<label class="f" style="margin-top:12px">现象描述<textarea name="description"></textarea></label><label class="f" style="margin-top:12px">关键数据（每行 指标=数值）<textarea name="data" placeholder="CP良率=86%"></textarea></label>' +
        '<div style="margin-top:12px;display:flex;gap:12px;align-items:center"><button class="btn" type="submit">创建</button><label><input type="checkbox" name="run" checked> 创建后立即分析</label><span class="muted">手工建案没有 MES/YMS 数据，结论主要依据已学习知识</span></div></form></div>' +
        '<div class="panel" style="margin-top:16px">' + caseTable(cases) + "</div>";
      bindRows();
      var go = function () { var p = new URLSearchParams(); [["f-st", "stage"], ["f-sy", "symptom"], ["f-ss", "status"], ["f-q", "q"]].forEach(function (a) { var v = document.getElementById(a[0]).value; if (v) p.set(a[1], v); }); location.hash = "#/cases" + (p.toString() ? "?" + p : ""); };
      ["f-st", "f-sy", "f-ss"].forEach(function (id) { document.getElementById(id).onchange = go; });
      document.getElementById("f-q").onkeydown = function (e) { if (e.key === "Enter") go(); };
      document.getElementById("new-btn").onclick = function () { var f = document.getElementById("new-form"); f.hidden = !f.hidden; };
      document.getElementById("cf").onsubmit = function (e) {
        e.preventDefault(); var f = new FormData(e.target), data = {};
        String(f.get("data") || "").split("\n").forEach(function (l) { var i = l.search(/[=:：]/); if (i > 0) data[l.slice(0, i).trim()] = l.slice(i + 1).trim(); });
        post("cases", { title: f.get("title"), description: f.get("description"), symptom: f.get("symptom") || null, severity: f.get("severity"), product: f.get("product"), site: f.get("site"), equipment: f.get("equipment"), cost_impact: parseFloat(f.get("cost_impact") || "0"), data: data, source: "dashboard" })
          .then(function (c) { location.hash = "#/cases/" + c.id + (f.get("run") ? "?run=1" : ""); }).catch(function (err) { toast(err.message); });
      };
    });
  }

  // ── 案例详情 ───────────────────────────────────────────
  function pageCase(id, params) {
    return api("cases/" + id).then(function (d) {
      var c = d.case, an = d.analyses[0], ds = d.dataset, ev = an ? an.evidence : (c.evidence || {}), cf = an ? an.counterfactual : null;
      var h = '<div class="page-head"><a href="#/cases" class="muted">← 案例库</a><h1>' + esc(c.code) + " " + esc(c.title) + "</h1>" + sevBadge(c.severity) + statusBadge(c.status) +
        '<span class="badge">' + esc(lbl("stages", c.stage)) + '</span><span class="badge accent">' + esc(symName(c.symptom)) + '</span><span class="spacer"></span>' +
        (c.status !== "closed" ? '<button class="btn" id="run-agent">⚙️ ' + (an ? "重新分析" : "运行分析工作流") + "</button>" : "") + "</div>";
      if (LAST_CLOSE[id]) h += learnedBox(LAST_CLOSE[id]);
      h += '<div class="grid g-side"><div class="grid">';
      h += '<div class="panel"><h2>现象与关键数据</h2><p>' + esc(c.description || "（无描述）") + '</p><dl class="kv">' +
        [["产品", c.product], ["工厂", c.site], ["来源", c.source], ["损失", c.cost_impact + " 万元"], ["创建", dt(c.created_at)]].filter(function (x) { return x[1]; }).map(function (x) { return "<dt>" + x[0] + "</dt><dd>" + esc(x[1]) + "</dd>"; }).join("") +
        Object.keys(c.data || {}).map(function (k2) { return "<dt>" + esc(k2) + '</dt><dd class="num"><b>' + esc(c.data[k2]) + "</b></dd>"; }).join("") + "</dl></div>";
      h += '<div class="panel" id="run-panel" hidden><h2>分析工作流（DBOS 持久化执行）</h2><div id="wf-id" class="muted steps-trace"></div><ul class="steps" id="steps"></ul></div>';
      if (ds) {
        h += '<div class="panel"><h2>统计证据 <span class="muted" style="font-weight:400;font-size:12px">MES 批次 ' + ds.lots + " 个 · 失效 " + ds.fails + " 个 · " + esc(ds.metric || "") + "</span></h2>";
        if (ds.wafers && ds.wafers.length) h += "<h3>晶圆图（最差晶圆 vs 正常晶圆）" + (ev.wafer_pattern ? ' · 识别为 <b>' + esc(ev.wafer_pattern) + "</b>（" + pct(ev.wafer_pattern_conf) + "）" : "") + '</h3><div class="wafers">' +
          ds.wafers.map(function (w) { var sig = (ev.wafer_signatures || []).find(function (s) { return s.wafer === w.wafer; }); return '<div class="wafer">' + waferSVG(w.map) + "<b>" + esc(w.wafer.slice(-9)) + "</b> " + (w.fail ? (sig ? esc(sig.pattern) : "失效") : "正常") + "</div>"; }).join("") + "</div>";
        h += "<h3>批次指标时间序列</h3>" + seriesChart(ds);
        var sus = (ev.suspects || []).filter(function (s) { return s.fails > 0; });
        h += "<h3>共性分析（Fisher 精确检验 + Bonferroni 校正）</h3>";
        if (!sus.some(function (s) { return s.p_adj < 0.05; })) h += '<div class="callout warn">没有任何机台/腔体/物料批次的失效富集达到统计显著 —— 证据模糊，排序将更多依赖从本厂历史案例学到的知识。</div>';
        if (sus.length) h += '<div class="table-wrap"><table><thead><tr><th>工序 · 机台/批次</th><th>批次</th><th>失效率 vs 其余</th><th>提升</th><th>p_adj</th></tr></thead><tbody>' +
          sus.slice(0, 5).map(function (s) { var sig = s.p_adj < 0.05; return '<tr class="' + (sig ? "" : "muted") + '"><td>' + esc(lbl("steps", s.step)) + " · <b>" + esc(s.value) + "</b></td><td>" + s.n + "</td><td>" + pct(s.rate_with) + " vs " + pct(s.rate_without) + "</td><td>" + s.lift + "×</td><td>" + s.p_adj + (sig ? ' <span class="badge warn">显著</span>' : " 不显著") + "</td></tr>"; }).join("") + "</tbody></table></div>";
        if (ev.change_point) h += '<div class="callout">变点：' + esc(ev.change_point.value) + " 自 " + dt(ev.change_point.since) + " 起失效率 " + pct(ev.change_point.rate_before) + " → " + pct(ev.change_point.rate_after) + (ev.change_point.after_pm ? "，<b>紧随 PM</b>" : "") + "</div>";
        if (ds.fdc && ds.fdc.length) h += "<h3>FDC · " + esc(ds.fdc_tool) + ((ev.fdc || []).length ? "（" + ev.fdc.map(function (f) { return esc(f.param) + " " + (f.z > 0 ? "+" : "") + f.z + "σ"; }).join("，") + "）" : "") + '</h3><div class="grid g2">' + fdcChart(ds.fdc) + "</div>";
        h += "</div>";
      }
      if (an) {
        var crit = an.critique || {}, rootCats = an.hypotheses.slice(0, 1).map(function (x) { return x.six_m; });
        h += '<div class="panel"><div class="page-head" style="margin-bottom:8px"><h2 style="margin:0">贝叶斯根因排序 v' + an.version + '</h2><span class="badge ' + ((crit.score || 0) >= 75 ? "ok" : "warn") + '">评审 ' + (crit.score == null ? "-" : crit.score) + '/100</span><span class="badge">' + esc(an.model) + '</span><span class="spacer"></span><span class="muted">' + dt(an.created_at) + "</span></div>" +
          (cf && cf.changed ? '<div class="callout ok">🧠 <b>持续学习改变了结论</b>：若只用教科书先验，首位会是「' + esc(cf.top_name) + "」（" + pct(cf.top_posterior) + "）；结合本厂已验证案例，首位为「" + esc(an.hypotheses[0].name) + "」（" + pct(an.hypotheses[0].posterior) + "）。</div>"
            : cf ? '<div class="callout">🧠 持续学习的影响：首位一致，概率 ' + pct(cf.learned_top_posterior_if_frozen) + " → " + pct(an.hypotheses[0].posterior) + "；不确定性 " + cf.entropy_frozen + " → " + cf.entropy_learned + " bit</div>" : "") +
          hypList(an.hypotheses, cf) +
          (cf && cf.tests && an.hypotheses[0].posterior < 0.6 ? '<div class="callout warn">🔬 根因尚不确定：建议优先鉴别「' + esc(cf.tests[0].name) + "」（预期信息增益 " + cf.tests[0].eig + " bit）</div>" : "") +
          (cf && cf.investigation ? '<div class="callout">🧪 代码模式调查员：' + cf.investigation.findings.map(esc).join("；") + "</div>" : "") +
          "<h3>5-Why</h3><ol class=\"whys\">" + an.five_whys.map(function (w) { return "<li>" + esc(w) + "</li>"; }).join("") + "</ol>" +
          "<h3>鱼骨图（6M，★ 为首位根因所在维度）</h3>" + fishbone(an.ishikawa || {}, rootCats, symName(c.symptom)) +
          '<h3>这份分析有用吗？</h3><div class="stars" data-aid="' + an.id + '">' + [1, 2, 3, 4, 5].map(function (n) { return '<button data-n="' + n + '" aria-label="' + n + ' 星">★</button>'; }).join("") + "</div>" +
          '<details style="margin-top:12px"><summary>完整 8D 报告</summary>' + md(an.report_md) + "</details></div>";
        h += harnessPanel(cf && cf.harness, an.hypotheses);
      } else h += '<div class="panel empty">尚未分析。点击右上角「运行分析工作流」，或在左侧对话输入「分析 ' + esc(c.code) + "」。</div>";
      h += '</div><div class="grid" style="align-content:start">';
      if (c.status !== "closed") {
        var opts = (an ? an.hypotheses.map(function (x) { return [x.cause_code, x.name + "（" + pct(x.posterior) + "）"]; }) : []);
        d.candidates.forEach(function (x) { if (!opts.some(function (o) { return o[0] === x.code; })) opts.push([x.code, x.name]); });
        h += '<div class="panel"><h2>确认根因并关案</h2><p class="muted" style="font-size:12px">关案 = 给系统一个经现场验证的标签：知识图谱与经验手册立即更新，学习曲线重新回放。</p>' +
          '<select id="cause-sel" style="width:100%">' + opts.map(function (o) { return '<option value="' + o[0] + '">' + esc(o[1]) + "</option>"; }).join("") + '<option value="__new">＋ 新根因（图谱中没有）</option></select>' +
          '<div id="new-cause" hidden style="margin-top:8px"><input id="nc-code" placeholder="根因代码，如 HBM_MUF_FLOW" style="width:100%;margin-bottom:6px"><input id="nc-name" placeholder="根因描述" style="width:100%"></div>' +
          '<button class="btn" id="close-btn" style="margin-top:10px;width:100%">确认并学习</button></div>';
      }
      if (d.learning_events.length) h += '<div class="panel"><h2>本案带来的学习</h2><ul class="timeline">' + d.learning_events.map(eventLine).join("") + "</ul></div>";
      h += '<div class="panel"><h2>CAPA 措施（' + d.actions.length + "）</h2>" + (d.actions.length ? d.actions.map(actionCard).join("") : '<div class="empty">分析后自动生成</div>') + '<p class="muted" style="font-size:12px">标记为“验证无效”会把该措施针对的根因记为被现场否定（负向学习）。</p></div>';
      var wf = an && an.trace.length ? an.trace[an.trace.length - 1].workflow_id : null;
      if (wf) h += '<div class="panel"><h2>持久化执行轨迹</h2><div class="muted steps-trace">' + esc(wf) + '</div><div id="wf-steps" class="steps-trace">加载中…</div></div>';
      h += "</div></div>";
      view.innerHTML = h;
      bindActionSelects();
      var rb = document.getElementById("run-agent"); if (rb) rb.onclick = function () { runAgent(c.id); };
      var sel = document.getElementById("cause-sel");
      if (sel) { sel.onchange = function () { document.getElementById("new-cause").hidden = sel.value !== "__new"; };
        document.getElementById("close-btn").onclick = function (e) {
          var body = sel.value === "__new" ? { cause_code: document.getElementById("nc-code").value.trim().toUpperCase(), root_cause: document.getElementById("nc-name").value.trim() } : { cause_code: sel.value };
          e.target.disabled = true;
          post("cases/" + c.id + "/close", body).then(function (r) { LAST_CLOSE[c.id] = r; toast("已关案并完成学习"); route(); document.getElementById("page-pane").scrollTop = 0; }).catch(function (err) { toast(err.message); e.target.disabled = false; });
        }; }
      if (wf) api("workflows/" + encodeURIComponent(wf) + "/steps").then(function (st) { document.getElementById("wf-steps").innerHTML = "<ol>" + st.map(function (s) { return "<li>" + esc(s.name) + ' <span class="muted">' + (s.ms == null ? "" : s.ms + " ms") + "</span>" + (s.error ? ' <span class="badge warn">失败</span>' : " ✔") + "</li>"; }).join("") + '</ol><p class="muted">每个步骤的输出都已检查点到 DBOS 系统库，进程崩溃后从断点继续。</p>'; }).catch(function () {});
      var stars = view.querySelector(".stars");
      if (stars) { var mine = d.feedback.filter(function (f) { return f.analysis_id === an.id; }).pop(); var paint = function (n) { stars.querySelectorAll("button").forEach(function (b) { b.classList.toggle("on", +b.dataset.n <= n); }); }; if (mine) paint(mine.rating);
        stars.querySelectorAll("button").forEach(function (b) { b.onclick = function () { post("feedback", { analysis_id: an.id, rating: +b.dataset.n }).then(function () { paint(+b.dataset.n); toast("感谢反馈"); }); }; }); }
      if (new URLSearchParams(params || "").get("run") && !an) runAgent(c.id);
    });
  }

  function learnedBox(r) {
    var s = r.learning_summary || {};
    return '<div class="panel" style="border-color:var(--ok);margin-bottom:16px"><h2>🧠 本次学到了什么</h2><p>确认根因「<b>' + esc(r.case.root_cause) + "</b>」" + (r.rank_in_analysis ? "（分析时排第 " + r.rank_in_analysis + " 位）" : "（分析时未列出，作为新知识加入）") + '</p><ul class="timeline">' +
      r.learning_events.map(eventLine).join("") + '</ul><div class="callout ok">学习曲线已重新回放（' + s.cases + " 个已验证案例）：首位命中率 持续学习 <b>" + pct(s.hit1_learning) + "</b> vs 冻结先验 " + pct(s.hit1_frozen) + ' · <a href="#/evolution?tab=continual">查看曲线 →</a></div></div>';
  }

  function actionCard(a) {
    var od = a.status !== "done" && a.status !== "verified" && a.status !== "ineffective" && a.due_date && new Date(a.due_date) < new Date();
    return '<div class="card' + (od ? " overdue" : "") + '"><div><span class="badge accent">' + esc(lbl("action_kinds", a.kind)) + "</span> " + esc(a.title) + "</div>" +
      '<div class="meta">' + (a.case_code ? '<a href="#/cases/' + a.case_id + '">' + esc(a.case_code) + "</a>" : "") + "<span>👤 " + esc(a.owner_role || "-") + "</span><span>📅 " + d10(a.due_date) + "</span>" + (od ? '<span class="badge warn">逾期</span>' : "") + "</div>" +
      (a.verify_metric ? '<div class="meta">验证：' + esc(a.verify_metric) + (a.target ? " → " + esc(a.target) : "") + "</div>" : "") + (a.result ? '<div class="meta">结果：' + esc(a.result) + "</div>" : "") +
      '<select data-act="' + a.id + '" aria-label="更新状态">' + Object.keys(META.action_statuses).map(function (s) { return '<option value="' + s + '"' + (s === a.status ? " selected" : "") + ">" + esc(META.action_statuses[s]) + "</option>"; }).join("") + "</select></div>";
  }
  function bindActionSelects() {
    view.querySelectorAll("select[data-act]").forEach(function (sel) {
      sel.onchange = function () {
        var body = { status: sel.value };
        if (sel.value === "verified" || sel.value === "ineffective") { var r = window.prompt("效果验证结果：", ""); if (r === null) { route(); return; } body.result = r; }
        api("actions/" + sel.dataset.act, { method: "PATCH", body: JSON.stringify(body) }).then(function () { toast(sel.value === "ineffective" ? "已记录：该根因被现场否定（负向学习）" : "已更新"); route(); }).catch(function (e) { toast(e.message); route(); });
      };
    });
  }

  function runAgent(id) {
    var panel = document.getElementById("run-panel"), steps = document.getElementById("steps"), btn = document.getElementById("run-agent");
    panel.hidden = false; steps.innerHTML = ""; btn.disabled = true; btn.textContent = "分析中…"; panel.scrollIntoView({ behavior: "smooth", block: "center" });
    api("cases/" + id + "/analyze?stream=true", { method: "POST", raw: true }).then(function (r) {
      var reader = r.body.getReader(), dec = new TextDecoder(), buf = "";
      function pump() { return reader.read().then(function (res) {
        if (res.done) return; buf += dec.decode(res.value, { stream: true }); var parts = buf.split("\n\n"); buf = parts.pop();
        parts.forEach(function (p) { if (p.indexOf("data:") !== 0) return; var ev = JSON.parse(p.slice(5));
          if (ev.type === "workflow") document.getElementById("wf-id").textContent = "workflow_id: " + ev.workflow_id;
          else if (ev.type === "step") steps.insertAdjacentHTML("beforeend", "<li><b>" + esc(ev.label) + "</b>" + esc(ev.detail || "") + "</li>");
          else if (ev.type === "done") { toast("分析完成" + (ev.learning_changed_top ? "：持续学习改变了首位根因" : "")); setTimeout(route, 700); }
          else if (ev.type === "error") toast(ev.message); });
        return pump(); }); }
      return pump();
    }).catch(function (e) { toast(e.message); btn.disabled = false; });
  }

  // ── CAPA ───────────────────────────────────────────────
  function pageCapa(params) {
    var overdue = new URLSearchParams(params || "").get("overdue") === "1";
    return api("actions" + (overdue ? "?overdue=true" : "")).then(function (acts) {
      view.innerHTML = '<div class="page-head"><h1>CAPA 看板</h1><span class="badge">' + acts.length + ' 条</span><span class="spacer"></span><a class="btn ' + (overdue ? "" : "ghost") + '" href="#/capa' + (overdue ? "" : "?overdue=1") + '">' + (overdue ? "显示全部" : "只看逾期") + "</a></div>" +
        '<div class="kanban">' + Object.keys(META.action_statuses).map(function (s) { var mine = acts.filter(function (a) { return a.status === s; }); return '<div class="col"><h4><span>' + esc(META.action_statuses[s]) + "</span><span>" + mine.length + "</span></h4>" + mine.slice(0, 40).map(actionCard).join("") + (mine.length > 40 ? '<div class="muted">… 另有 ' + (mine.length - 40) + " 条</div>" : "") + "</div>"; }).join("") + "</div>";
      bindActionSelects();
    });
  }

  function memoryTable(items) {
    if (!items.length) return '<div class="empty">暂无</div>';
    return '<div class="table-wrap"><table><thead><tr><th>类型</th><th>症状</th><th>条件 → 内容</th><th>通过</th><th>失败</th><th>来源</th><th>状态</th><th></th></tr></thead><tbody>' +
      items.map(function (b) {
        var src = { real: '<span class="badge ok">产线</span>', sandbox: '<span class="badge accent">孪生</span>', both: '<span class="badge ok">产线</span> <span class="badge accent">孪生</span>' }[b.source] || "";
        return "<tr><td>" + esc(lbl("memory_kinds", b.kind)) + "</td><td>" + esc(b.symptom_name) + "</td><td>" + esc(b.text) + '</td><td class="num">' + b.passes + '</td><td class="num">' + b.fails + "</td><td>" + src +
          '<div class="muted" style="font-size:11px">' + b.source_cases.slice(-3).map(function (id) { return '<a href="#/cases/' + id + '">' + code(id) + "</a>"; }).join(" ") + (b.source_tasks.length ? " 练习 " + b.source_tasks.length + " 题" : "") + "</div></td><td>" +
          (b.status === "active" ? '<span class="badge ok">生效</span>' : '<span class="badge">已停用</span>') + '</td><td><button class="btn sm ghost" data-mem="' + b.id + '" data-on="' + (b.status === "active" ? 0 : 1) + '">' + (b.status === "active" ? "停用" : "启用") + "</button></td></tr>";
      }).join("") + "</tbody></table></div>";
  }
  function bindMemoryButtons() {
    view.querySelectorAll("button[data-mem]").forEach(function (b) { b.onclick = function () { post("memory/" + b.dataset.mem + "/status", { active: b.dataset.on === "1" }).then(function () { route(); }); }; });
  }

  // ── AG-UI 客户端（零依赖：fetch + SSE 解析） ─────────────
  function agui(body, onEvent) {
    var headers = { "Content-Type": "application/json", Accept: "text/event-stream" };
    if (key()) headers.Authorization = "Bearer " + key();
    return fetch("agui", { method: "POST", headers: headers, body: JSON.stringify(body) }).then(function (r) {
      if (!r.ok) throw new Error("AG-UI 请求失败 " + r.status);
      var reader = r.body.getReader(), dec = new TextDecoder(), buf = "";
      function pump() { return reader.read().then(function (res) {
        if (res.done) return; buf += dec.decode(res.value, { stream: true }); var parts = buf.split("\n\n"); buf = parts.pop();
        parts.forEach(function (p) { var line = p.split("\n").filter(function (x) { return x.indexOf("data:") === 0; }).join(""); if (line) onEvent(JSON.parse(line.slice(5))); });
        return pump(); }); }
      return pump();
    });
  }
  function applyPatch(state, ops) { ops.forEach(function (op) { var parts = op.path.split("/").slice(1); var tgt = state; for (var i = 0; i < parts.length - 1; i++) tgt = tgt[parts[i]]; var k = parts[parts.length - 1]; if (op.op === "add" && Array.isArray(tgt) && k === "-") tgt.push(op.value); else tgt[k] = op.value; }); }
  // ── 自我进化：RSI 学习（数字孪生）+ 持续学习（产线），写入同一套记忆 ─────────
  var FACTOR_TIPS = { "先验知识": "本厂各根因有多常见（Dirichlet 先验）", "共性分析": "失效是否集中在某台机台 / 腔体 / 物料批次（Fisher 精确检验）",
    "FDC": "传感器参数的 σ 偏移是否吻合该根因", "PM 时序": "失效是否紧随某次保养", "机台热点": "这类根因是否常出在这台机台上" };
  var SRC = { rsi: ["accent", "孪生"], production: ["ok", "产线"], nightly: ["", "夜间"] };
  function srcBadge(s) { var x = SRC[s] || ["", s]; return '<span class="badge ' + x[0] + '">' + esc(x[1]) + "</span>"; }

  function weightsCard(W) {
    var cur = W && W.current, f = (W && W.factors) || [];
    if (!cur) return '<div class="empty">还没有学到证据权重：需要至少 20 个已验证样本</div>';
    var max = Math.max.apply(null, f.map(function (k) { return cur.weights[k] == null ? 1 : cur.weights[k]; }).concat([2]));
    var g = cur.gate || {};
    return f.map(function (k) { var w = cur.weights[k] == null ? 1 : cur.weights[k];
      return '<div class="wbar"><span title="' + esc(FACTOR_TIPS[k] || "") + '">' + esc(k) + '</span><span class="track"><span class="fill" style="width:' + Math.max(2, w / max * 100) + '%"></span><span class="ref" style="left:' + (100 / max) + '%"></span></span><span class="num"><b>×' + w.toFixed(2) + "</b></span></div>"; }).join("") +
      '<p class="muted small">竖线 = 专家设定 ×1。×2 表示这类证据的说服力被学成专家设定的 2 倍（作为对数似然的指数）。当前版本 ' + srcBadge(cur.source) + " 基于 " + cur.n + " 个已验证样本" +
      (g.before ? "；验收门面板首位命中率 " + g.before.hit1 + " → <b>" + g.after.hit1 + "</b>，对数损失 " + g.before.logloss + " → " + g.after.logloss : "") + "。</p>" +
      "<h3>版本历史（只有通过验收门的版本才会保存）</h3><div class=\"chips\">" + W.history.slice(-14).map(function (h) { return '<span class="chip" title="' + esc(Object.keys(h.weights).map(function (k) { return k + " ×" + h.weights[k]; }).join("\n")) + '">' + esc((SRC[h.source] || ["", h.source])[1]) + " · n=" + h.n + " · " + dt(h.created_at).slice(5) + "</span>"; }).join("") + "</div>";
  }

  var MODE_LABEL = { none: "不做 RSI", brs: "仅广度 BRS", drs: "仅深度 DRS", full: "论文方案 BRS→DRS（v0.3）", full_w: "＋ 证据权重学习", v2: "＋ 证据校准（v2，默认）" };
  function benchBars(b) {
    if (!b) return '<div class="empty">暂无</div>';
    var modes = ["none", "brs", "drs", "full", "full_w", "v2"].filter(function (m) { return b.modes[m]; });
    var orc = b.oracle ? b.oracle.hit1 : null;
    var lo = Math.floor(Math.min.apply(null, modes.map(function (m) { return b.modes[m].hit1_min; })) / 5) * 5 - 5;
    var hi = Math.ceil(Math.max.apply(null, modes.map(function (m) { return b.modes[m].hit1_max; }).concat(orc ? [orc] : [])) / 5) * 5 + 3;
    var X = function (v) { return (v - lo) / (hi - lo) * 100; };
    return modes.map(function (m) { var v = b.modes[m], best = m === "v2";
      return '<div class="bench"><div class="bench-h"><b>' + MODE_LABEL[m] + '</b><span class="num"><b>' + v.hit1_mean + "</b></span></div>" +
        '<div class="bench-t"><div style="width:' + X(v.hit1_mean) + "%;background:" + (best ? "var(--chart-1)" : "var(--chart-2)") + '"></div>' +
        '<i title="最差 ' + v.hit1_min + " · 最好 " + v.hit1_max + '" style="left:' + X(v.hit1_min) + "%;width:" + Math.max(0.8, X(v.hit1_max) - X(v.hit1_min)) + '%"></i>' +
        (orc ? '<u style="left:' + X(orc) + '%" title="参考天花板 ' + orc + '"></u>' : "") + "</div>" +
        '<div class="muted small">种子范围 ' + v.hit1_min + "–" + v.hit1_max + " · 真因后验 " + v.partial_mean + " · 平均 " + v.tasks_mean + " 道练习</div></div>"; }).join("") +
      '<p class="muted small">横轴 ' + lo + "–" + hi + "（留出测试集首位命中率）。条长 = " + b.seeds.length + " 个随机种子的均值；方框 = 最差到最好；" + (orc ? "红色虚线 = 参考天花板 " + orc + "（知道孪生真实先验、且证据权重直接在测试集上拟合时的命中率，任何学习都不可能稳定超过）；" : "") +
      "测试集 " + b.eval_size + " 题，按本厂真实根因分布分层抽样。</p>";
  }

  function rsiCurve(points, opts) {
    opts = opts || {};
    if (!points || points.length < 2) return '<div class="empty">暂无</div>';
    var W = 680, H = opts.h || 230, pl = 36, pr = 60, pt = 14, pb = 28, n = points.length, maxStep = points[n - 1].step || 1;
    var vals = points.map(function (p) { return p.hit1; }).concat(points.map(function (p) { return p.partial; }));
    var lo = Math.max(0, Math.floor(Math.min.apply(null, vals) / 10) * 10 - 5), hi = Math.min(100, Math.ceil(Math.max.apply(null, vals) / 10) * 10 + 5);
    var x = function (st) { return pl + st / maxStep * (W - pl - pr); }, y = function (v) { return pt + (1 - (v - lo) / (hi - lo)) * (H - pt - pb); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="自我进化：练习题数与留出测试集得分">', start = 0;
    ["BRS", "CAL", "DRS"].forEach(function (ph, i) { var end = points.filter(function (p) { return p.phase === ph; }).map(function (p) { return p.step; }).pop();
      if (end == null) return; s += '<rect x="' + x(start) + '" y="' + pt + '" width="' + Math.max(1, x(end) - x(start)) + '" height="' + (H - pt - pb) + '" fill="' + (i === 1 ? "var(--ok-soft)" : "var(--accent-soft)") + '" opacity="' + (i === 2 ? 0.35 : 0.7) + '"/><text x="' + (x(start) + 4) + '" y="' + (pt + 11) + '">' + { BRS: "广度", CAL: "证据校准", DRS: "深度" }[ph] + "</text>"; start = end; });
    [lo, (lo + hi) / 2, hi].forEach(function (g) { s += '<line class="axis" x1="' + pl + '" x2="' + (W - pr) + '" y1="' + y(g) + '" y2="' + y(g) + '"/><text x="' + (pl - 4) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + Math.round(g) + "</text>"; });
    var line = function (k) { return points.map(function (p) { return x(p.step).toFixed(1) + "," + y(p[k]).toFixed(1); }).join(" "); };
    s += '<polyline fill="none" stroke="var(--chart-1)" stroke-width="2.6" points="' + line("hit1") + '"/><polyline fill="none" stroke="var(--chart-line)" stroke-width="2" stroke-dasharray="5 4" points="' + line("partial") + '"/>';
    var last = points[n - 1];
    s += '<text x="' + (x(last.step) + 6) + '" y="' + (y(last.hit1) + 4) + '" style="fill:var(--chart-1);font-weight:700">' + last.hit1 + '</text><text x="' + (x(last.step) + 6) + '" y="' + (y(last.partial) + 4) + '" style="fill:var(--chart-line)">' + last.partial + "</text>";
    points.forEach(function (p) { s += '<circle cx="' + x(p.step) + '" cy="' + y(p.hit1) + '" r="5" fill="transparent"><title>第 ' + p.step + " 题后（" + p.phase + "）\n首位命中 " + p.hit1 + "\n真因后验 " + p.partial + "\n生效记忆 " + p.memory + "</title></circle>"; });
    s += '<text x="' + x(0) + '" y="' + (H - 8) + '">0（不做 RSI）</text><text x="' + x(maxStep) + '" y="' + (H - 8) + '" text-anchor="end">第 ' + maxStep + " 题</text>";
    return s + '</svg><div class="legend"><span><i style="background:var(--chart-1)"></i>首位命中率</span><span><i style="background:var(--chart-line)"></i>真因后验（虚线）</span></div>';
  }

  function sourcesDiagram() {
    return '<div class="flow"><div class="flow-src"><div class="flow-box ok"><b>① 持续学习（产线）</b><span>工程师确认根因 / CAPA 验证有效或无效</span><span class="muted small">真实标签 · 权重 1.0 · 随时发生</span></div>' +
      '<div class="flow-box accent"><b>② RSI 学习（数字孪生）</b><span>课程 agent 出题 → 执行 agent 作答 → 孪生干预反馈 → 验证 agent 判定</span><span class="muted small">孪生标签 · 权重 0.5 · 按需 / 每晚</span></div></div>' +
      '<div class="flow-arrow"><b class="arr">→</b><span class="muted small">只有验证过的结论<br>才能写入</span></div>' +
      '<div class="flow-box mem"><b>同一套可进化记忆</b><span>根因先验（本厂常见什么）</span><span>机台热点（常出在哪台）</span><span><b>证据权重</b>（每类证据该信多少）· v2 新增</span><span>流程与教训（带条件与通过/失败次数）</span>' +
      '<span class="muted small">每次更新证据权重都要过验收门：在封存面板上不退步才采纳</span></div></div>';
  }

  function practiceTable(tasks) {
    var main = tasks.filter(function (t) { return t.phase !== "CAL"; }), cal = tasks.filter(function (t) { return t.phase === "CAL"; });
    var calPass = cal.filter(function (t) { return t.verdict === "PASS"; }).length;
    return '<div class="table-wrap"><table><thead><tr><th>#</th><th>阶段</th><th>课程 agent 出题</th><th>执行 agent 作答（依次尝试）</th><th>验证 agent 依据</th><th>判定</th><th>记忆变化</th><th>孪生真因（仅事后审计）</th></tr></thead><tbody>' +
      main.map(function (t) {
        var att = t.attempts.map(function (a, i) { return (i + 1) + ". " + esc(String(a.cause || a.cause_code).slice(0, 14)) + (a.posterior != null ? "（" + pct(a.posterior) + "）" : "") + (a.verdict.passed ? " ✔" : " ✘"); }).join("<br>");
        var last = t.attempts[t.attempts.length - 1] || { verdict: {} };
        var mem = (t.memory_changes || []).filter(function (e) { return ["memory_add", "memory_pass", "memory_fail", "hotspot", "edge_added", "memory_retire"].indexOf(e.kind) >= 0; }).map(function (e) { return esc(e.detail.slice(0, 26)); }).join("<br>");
        return '<tr><td class="num">' + t.step + "</td><td>" + esc(t.phase) + "</td><td><b>" + esc(symName(t.spec.symptom)) + '</b><br><span class="badge">' + esc(t.spec.variant) + "</span> " + esc(t.spec.difficulty) + (t.spec.decoy_pm ? " + PM 干扰" : "") +
          '<div class="muted small">' + esc(t.spec.rationale) + "</div></td><td>" + att + '</td><td class="muted small">' + (last.verdict.evidence || []).map(esc).join("<br>") + "</td><td>" + (t.verdict === "PASS" ? '<span class="badge ok">PASS</span>' : '<span class="badge warn">FAIL</span>') +
          '</td><td class="small">' + (mem || '<span class="muted">-</span>') + '</td><td class="muted small">' + esc(t.hidden.cause || "") + (t.hidden.tool ? " @ " + esc(t.hidden.tool) : "") + "</td></tr>"; }).join("") +
      (cal.length ? '<tr><td></td><td>CAL</td><td colspan="6"><b>证据校准 ' + cal.length + " 道</b>（验证通过 " + calPass + " 道）：难度易 / 中 / 难 = 2 : 4 : 4，按信息价值分配症状；每题最多试 5 个假设以减少选择偏差。<span class=\"muted\">在线模式下只用来学习证据权重，不改动产线先验；产线变更涉及的症状除外（孪生已同步该变更）。</span></td></tr>" : "") +
      "</tbody></table></div>";
  }

  function liveBox() {
    if (!LIVE) return "";
    var st = LIVE.state || {};
    return '<div class="panel live" id="live-box"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">实时直播 · 会话 #' + esc(st.session_id) + '</h2><span class="badge ' + (LIVE.done ? "ok" : "accent") + '">' + (LIVE.done ? "已完成" : "进行中") + '</span><span class="muted small">AG-UI 事件流：STEP · SUBAGENT · STATE_DELTA · CUSTOM</span></div>' +
      rsiCurve(st.curve || [], { h: 180 }) + '<ul class="timeline" style="max-height:220px;overflow:auto">' + LIVE.log.slice(-40).map(function (l) { return '<li class="' + (l[1] || "") + '">' + l[0] + "</li>"; }).join("") + "</ul></div>";
  }
  function renderLive() { var el = document.getElementById("live-box"); if (el) el.outerHTML = liveBox(); }

  var TRIGGER = { schedule: "每月定时", seed: "启动时", console: "对话", manual: "手动" };
  function trigName(t) { return TRIGGER[t] || String(t || "").replace(/^user:/, ""); }
  function hitPlain(s) { if (!s || s.hit1_before == null) return "-"; return s.plateau ? s.hit1_after + "（持平）" : s.hit1_before + " → " + s.hit1_after; }
  function plateauWhy(s) { return s.plateau_reason === "no_gap" ? "这次产线变更没有让命中率下降，没有需要追赶的差距" : "上次会话之后孪生没有新的产线变更，也没有发现新的知识缺口，记忆已经跟上产线现状"; }
  function hitCell(s) {
    if (!s || s.hit1_before == null) return '<span class="muted">进行中</span>';
    if (s.plateau) return "<b>" + s.hit1_after + '</b> <span class="badge" title="' + plateauWhy(s) + '">持平</span>';
    var d = Math.round((s.hit1_after - s.hit1_before) * 100) / 100;
    return s.hit1_before + " → <b>" + s.hit1_after + '</b> <span class="badge ' + (d > 0 ? "ok" : "warn") + '">' + (d > 0 ? "+" : "") + d + "</span>";
  }
  function changeText(cfg) { var ch = (cfg && cfg.changes) || []; return ch.length ? ch.map(function (c) { return esc(c.label); }).join("；") : '<span class="muted">无变更</span>'; }
  function sessionTable(sessions, sid) {
    return '<div class="table-wrap"><table><thead><tr><th>#</th><th>时间</th><th>触发</th><th>上次会话之后的产线变更</th><th>留出集首位命中率</th><th>练习题</th><th>权重更新 采纳 / 拒绝</th></tr></thead><tbody>' +
      sessions.map(function (x) { var sm = x.summary || {};
        return '<tr class="click' + (x.id === sid ? " sel" : "") + '" data-href="#/evolution?tab=rsi&s=' + x.id + '"><td class="num">' + x.id + '</td><td class="num">' + d10(x.created_at) + "</td><td>" + esc(trigName(x.trigger)) + '</td><td class="small">' + changeText(x.config) + '</td><td class="num">' + hitCell(sm) + '</td><td class="num">' + (sm.tasks || "-") + '</td><td class="num">' + (sm.refits_accepted == null ? "-" : sm.refits_accepted + " / " + sm.refits_rejected) + "</td></tr>"; }).join("") + "</tbody></table></div>";
  }
  // ── Harness：快慢双脑 · 慢脑探测裁决 · 改进经验库 · RRSI 进化 harness 组件 ─────────
  var REASON_ZH = { missing_data: "缺少批次数据", critical: "S1 重大", first_seen: "无先例", prior_error: "同条件快脑答错过", audit: "周期抽查",
    low_conf: "校准概率不足", ambiguous: "集合不止一个根因", rule: "条件规则" };
  var PARAM_ZH = { "router.fast_min_conf": "快脑门槛", "router.audit_every": "抽查间隔", "slow.ttt": "近邻训练",
    "slow.ttt_tasks": "近邻题量", "slow.ttt_keep": "旧份额保留", "arbiter.min_gain": "改判门槛", "experience.use": "按改进库调整",
    "investigator.mode": "大模型调查员", "router.force_slow": "强制慢脑" };
  var CHALLENGER = { fast: "快脑模型", ttt: "近邻训练", bayes: "贝叶斯意见" };
  var STATUS_ZH = { stable: "稳定", active: "生效", candidate: "候选", retired: "已停用" };
  function ruleText(r) {
    var c = Object.keys(r["if"] || {}).map(function (k) { var v = r["if"][k]; return k === "symptom" ? symName(v) : k === "evidence" ? (v === "strong" ? "共性显著" : "证据模糊") : k === "after_pm" ? (v ? "PM 后突变" : "与 PM 无关") : { tool: "富集在机台", batch: "富集在批次", none: "无富集" }[v] || v; }).join("·");
    var s = Object.keys(r.set || {}).map(function (k) { return (PARAM_ZH[k] || k) + (k === "router.force_slow" ? "" : (r.set[k] ? " 开" : " 关")); }).join("、");
    return "若［" + esc(c) + "］→ " + esc(s);
  }
  function editText(e) {
    if (e.component === "rule") return "<b>" + (e.op === "add" ? "加规则" : "删规则") + "</b> " + ruleText(e.rule) + (e.hypothesis ? '<div class="muted">' + esc(e.hypothesis) + "</div>" : "");
    return "<b>" + esc(PARAM_ZH[e.component] || e.component) + "</b> " + e.from + " → " + e.to + (e.hypothesis ? '<div class="muted">' + esc(e.hypothesis) + "</div>" : "");
  }
  function hitsTable(inner) {
    if (!inner || !inner.hits) return "";
    var keys = Object.keys(inner.hits), ch = (inner.arbiter || {}).chosen;
    return '<div class="table-wrap"><table><thead><tr><th>挑战者（同一批 ' + inner.probe_n + ' 道近邻探测题）</th><th>探测命中</th><th>对本案的首位</th></tr></thead><tbody>' +
      keys.map(function (k) { return "<tr" + (k === ch ? ' class="sel"' : "") + "><td>" + esc(CHALLENGER[k] || k) + (k === ch ? ' <span class="badge ok">采用</span>' : "") + '</td><td class="num"><b>' + inner.hits[k] + "</b> / " + inner.probe_n + "</td><td>" + esc(((inner.tops || {})[k]) || "-") + "</td></tr>"; }).join("") +
      '</tbody></table></div><p class="small">裁决：' + esc((inner.arbiter || {}).why || "") + "</p>";
  }
  function harnessPanel(hz, hyps) {
    if (!hz) return "";
    var names = {}; (hyps || []).forEach(function (h) { names[h.cause_code] = h.name; });
    var inner = hz.inner || {}, probes = inner.probes || [], top = (hyps || [])[0];
    var who = hz.fast_kind === "model" ? "快脑模型 v" + hz.fast_model : "快脑（贝叶斯，模型未晋升）";
    return '<div class="panel"><div class="page-head" style="margin-bottom:8px"><h2 style="margin:0">快慢双脑</h2>' +
      '<span class="badge ' + (hz.route === "fast" ? "ok" : "accent") + '">' + (hz.route === "fast" ? esc(who) + "：直接给结论" : "慢脑：探测题裁决") + "</span>" +
      '<span class="badge ' + (hz.status === "supported" ? "ok" : "warn") + '">' + (hz.status === "supported" ? "可直接执行" : "需现场鉴别") + "</span></div>" +
      '<div class="mini-stats"><span>首位是真因的校准概率 <b>' + pct(hz.reliability) + "</b></span><span>90% 把握的根因集合 <b>" + hz.set.map(function (c) { return esc(names[c] || c); }).join("、") + "</b></span></div>" +
      (top && top.model_why && top.model_why.length ? '<p class="small">快脑模型的主要依据（相对其它候选）：' + top.model_why.map(function (w) { return esc(w.name) + (w.delta > 0 ? " ↑" : " ↓") + Math.abs(w.delta).toFixed(2); }).join("；") + "</p>" : "") +
      (hz.reasons && hz.reasons.length ? '<p class="muted small">转慢脑原因：' + hz.reasons.map(esc).join("、") + "</p>" : "") +
      hitsTable(inner) +
      (probes.length ? '<p class="small">证据族对照：' + probes.map(function (p) { return "去掉" + esc(p.label) + (p.changed ? "后首位变为「" + esc(p.name) + "」" : "后首位不变"); }).join("；") + "</p>" : "") +
      ((hz.experiences || []).length ? "<h3>用到的改进经验</h3><ul class=\"small\">" + hz.experiences.map(function (x) { return "<li>" + esc(x.text) + ' <span class="muted">成功率 ' + pct(x.p) + " · " + x.n + " 次 · " + esc(STATUS_ZH[x.status] || x.status) + "</span></li>"; }).join("") + "</ul>" : "") +
      "</div>";
  }
  function ciText(d) { return d && d.ci95 ? "[" + (d.ci95[0] * 100).toFixed(1) + ", " + (d.ci95[1] * 100).toFixed(1) + "] 个百分点" : "-"; }
  function benchTable(b) {
    if (!b) return '<div class="empty">还没有孪生对照结果</div>';
    var r = b.results, row = function (name, x, note) { return "<tr><td>" + name + '</td><td class="num"><b>' + x.hit1 + '</b></td><td class="num">' + pct(x.fast_rate) + '</td><td class="num">' + x.fast_wrong + '</td><td class="num">' + (x.fast_precision == null ? "-" : pct(x.fast_precision)) + '</td><td class="num">' + pct(x.coverage) + '</td><td class="num">' + (x.cost_per_task == null ? "-" : x.cost_per_task) + '</td><td class="muted small">' + (note || "") + "</td></tr>"; };
    var s = r.stress;
    return '<div class="table-wrap"><table><thead><tr><th>同一批孪生 report 题</th><th>首位命中率</th><th>快脑接手</th><th>快脑答错</th><th>快脑准确率</th><th>保形覆盖</th><th>每题孪生仿真</th><th></th></tr></thead><tbody>' +
      row("只用贝叶斯", r.bayes_only, "统计内核 + 记忆，全部直接给结论") + row("只用快脑模型", r.model_only, "对贝叶斯配对差 " + ciText(r.model_vs_bayes)) +
      row("快慢双脑", r.dual, "对只用快脑模型配对差 " + ciText(r.dual_vs_model)) + "</tbody></table></div>" +
      '<div class="callout ' + (s.diff.ci95[0] > 0 ? "ok" : "") + '"><b>压力场景：产线刚变更，快脑模型和记忆都还没跟上</b>（' + s.symptoms.map(symName).map(esc).join("、") + "：" + esc(s.change || "") + "）<br>只用贝叶斯 <b>" + s.bayes_only.hit1 + "</b> · 只用快脑模型 <b>" + s.model_only.hit1 + "</b> → 快慢双脑 <b>" + s.dual.hit1 + "</b>（对只用快脑 " + ciText(s.diff) + "；慢脑改判率 " + pct(s.dual.overrule_rate) + "）</div>" +
      '<p class="muted small">report 题与 RRSI 搜索、选择用的题互不重叠。平常场景下快脑模型已经跟上产线，慢脑的价值主要是“哪些结论可以直接信”；近邻训练的价值在产线变更之后、下一个训练窗口之前这段时间。</p>';
  }
  function harnessFlow() {
    return '<div class="flow"><div class="flow-src"><div class="flow-box ok"><b>快脑（System 1）= 训练出来的小模型</b><span>Wide &amp; Deep &amp; Cross 排序网络，毫秒级给根因排序和校准概率</span><span class="muted small">概率够高、保形集合只有首位、改进库没报过错 → 直接给结论</span></div>' +
      '<div class="flow-box accent"><b>慢脑（System 2）· 双层递归的内层</b><span>挑战者：近邻训练（临时记忆在近邻题上学，快脑模型冻结参数读它重排）· 贝叶斯意见</span><span>同一批近邻探测题上比命中，明显赢过快脑才改判；有 Key 时大模型调查员再核查</span></div></div>' +
      '<div class="flow-arrow"><b class="arr">→</b><span class="muted small">外层<br>经验反思</span></div>' +
      '<div class="flow-box mem"><b>改进经验库（Harness²）</b><span>在什么条件下快脑可信 / 哪个挑战者赢得过快脑</span><span>Beta 账本记成败；答错追责，失败多了停用（留档）</span>' +
      '<span class="muted small">RRSI：有触发才从改进库提议修改基础 harness（组件 + 条件规则）；慢脑推翻快脑的次数会触发快脑训练窗口</span></div></div>';
  }
  function pageHarness(head) {
    return api("harness").then(function (o) {
      var r = o.receipts, ex = o.experiences;
      var tiles = [[pct(r.total ? r.fast / r.total : 0), "快脑接手（共 " + r.total + " 次分析）"],
        [r.fast_verified ? r.fast_correct + "/" + r.fast_verified : "-", "快脑结论经现场确认正确" + (r.fast_precision_ci95 ? "（95% 区间 " + pct(r.fast_precision_ci95[0]) + "–" + pct(r.fast_precision_ci95[1]) + "）" : "")],
        [r.overrule_verified ? r.overrule_correct + " vs " + r.overrule_fast_correct : "-", "慢脑改判且已确认：改判后对 vs 快脑原本对（共 " + r.overrule_verified + " 例）"],
        [r.coverage == null ? "-" : pct(r.coverage), "保形集合覆盖现场真因（目标 90%）"],
        [r.with_model + "/" + r.total, "用训练出来的快脑模型（其余在模型晋升前用贝叶斯）"]];
      var months = r.by_month.slice(-14), mx = Math.max.apply(null, months.map(function (m) { return m.fast + m.slow; }).concat([1]));
      view.innerHTML = head + '<div class="tiles">' + tiles.map(function (t) { return '<div class="tile"><div class="v">' + esc(t[0]) + '</div><div class="l">' + esc(t[1]) + "</div></div>"; }).join("") + "</div>" +
        '<div class="panel">' + harnessFlow() + "</div>" +
        '<div class="panel" style="margin-top:16px"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">孪生对照：贝叶斯 vs 快脑模型 vs 快慢双脑</h2><span class="spacer"></span><button class="btn ghost sm" id="hz-bench">↻ 重跑对照</button></div>' + benchTable(o.benchmark) + "</div>" +
        '<div class="grid g2" style="margin-top:16px"><div class="panel"><h2>每月路由（去年至今）</h2><div class="route-bars">' + months.map(function (m) { return '<div class="rb"><span class="rb-t"><i class="f" style="height:' + (m.fast / mx * 100) + '%"></i><i class="s" style="height:' + (m.slow / mx * 100) + '%"></i></span><span class="muted small">' + esc(m.month.slice(2)) + "</span></div>"; }).join("") + '</div><div class="legend"><span><i style="background:var(--ok)"></i>快脑</span><span><i style="background:var(--accent)"></i>慢脑</span></div>' +
        "<h3>转慢脑的原因</h3><div class=\"chips\">" + Object.keys(r.reasons).sort(function (a, b) { return r.reasons[b] - r.reasons[a]; }).map(function (k) { return '<span class="chip">' + esc(REASON_ZH[k] || k) + " " + r.reasons[k] + "</span>"; }).join("") + "</div>" +
        "<h3>慢脑裁决结果</h3><div class=\"chips\">" + Object.keys(r.chosen).map(function (k) { return '<span class="chip">' + esc(CHALLENGER[k] || k) + " " + r.chosen[k] + "</span>"; }).join("") + "</div></div>" +
        '<div class="panel"><h2>改进经验库（Harness²）</h2><div class="chips">' + ["stable", "active", "retired"].map(function (s) { var n = ex.counts["tip:" + s]; return n ? '<span class="chip">' + STATUS_ZH[s] + " " + n + "</span>" : ""; }).join("") + "</div>" +
        '<ul class="timeline small" style="margin-top:8px;max-height:300px;overflow:auto">' + ex.items.slice(0, 14).map(function (x) { return '<li class="' + (x.status === "retired" ? "refute" : x.status === "stable" ? "confirm" : "") + '">' + esc(x.text) + ' <span class="muted">成功率 ' + pct(x.p) + " · " + x.n + " 次" + (x.retired_reason ? " · 停用：" + esc(x.retired_reason) : "") + "</span></li>"; }).join("") + "</ul>" +
        '<p class="muted small">产线事实（机台 ↔ 根因）在 RSIAgent 的 TTSE 事实轨里，这里只记 harness 组件“在什么条件下有效”。</p></div></div>' +
        '<div class="panel" style="margin-top:16px"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">RRSI：基础 harness 的配置存档（跨 run 继承）</h2><span class="muted small">待触发：' + (o.triggers.length ? o.triggers.map(esc).join("、") : "无（不定时，有信号才跑）") + '</span><span class="spacer"></span><button class="btn ghost sm" id="hz-evolve">▶ 手动跑一次</button></div>' +
        '<p class="small" style="margin-top:0">在用配置 ' + (o.policy.id ? "#" + o.policy.id : "默认") + "：" + Object.keys(o.policy.params).filter(function (k) { return k !== "rules"; }).map(function (k) { return esc(PARAM_ZH[k] || k) + " " + o.policy.params[k]; }).join(" · ") +
        ((o.policy.params.rules || []).length ? "<br>条件规则：" + o.policy.params.rules.map(ruleText).join("；") : "") + "</p>" +
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>时间</th><th>改动</th><th>选择集命中率 / 压力题</th><th>每题孪生仿真</th><th>结果</th></tr></thead><tbody>' +
        o.archive.slice(0, 16).map(function (p) { var sel = p.scores && p.scores.select; return "<tr><td class=\"num\">" + p.id + '</td><td class="num">' + d10(p.created_at) + '</td><td class="small">' + (p.edits || []).map(editText).join("") + '</td><td class="num">' + (sel ? sel.hit1 + (sel.hit1_stress != null ? " / " + sel.hit1_stress : "") : "-") + '</td><td class="num">' + (sel && sel.cost_per_task != null ? sel.cost_per_task : "-") + "</td><td>" + '<span class="badge ' + ({ active: "ok", accepted: "ok", rejected: "warn", rolled_back: "warn" }[p.status] || "") + '">' + esc({ active: "在用", accepted: "过选择门", rejected: "拒绝", retired: "已替换", rolled_back: "已回滚", candidate: "候选" }[p.status] || p.status) + '</span><div class="muted small">' + esc(p.note || "") + "</div></td></tr>"; }).join("") + "</tbody></table></div>" +
        '<p class="muted small"><b>何时跑</b>：没有存档 / 改进库新出现 ≥3 条稳定或停用的经验 / 产线确认的快脑错误 ≥3 次且 ≥15% / 新快脑模型上线 / 孪生同步了产线变更 / 慢脑赢过快脑的比例跌到两次前的 85% 以下；连续两次没接受任何候选就退避，只响应强信号。' +
        '<b>是否上线</b>：白名单与防泄漏审查 → search 初筛 → select 验收（按候选数做 Bonferroni 校正；噪声带内只接受更便宜或更简单的）→ 每次全新的 report 题 + 产线回放组成的上线门 → 观察期（错误率显著高于上一版就回滚）。<b>正则</b>：改动预算退火 3 → 2 → 1、复杂度感知、剪枝。</p></div>';
      var bind = function (id, path, msg) { var b = document.getElementById(id); if (b) b.onclick = function () { b.disabled = true; b.textContent = "运行中…"; post(path).then(function () { toast(msg); route(); }).catch(function (e) { toast(e.message); b.disabled = false; }); }; };
      bind("hz-bench", "harness/benchmark", "孪生对照已更新"); bind("hz-evolve", "harness/evolve", "RRSI 运行完成");
    });
  }
  // ── 快脑训练：A-RSI 训练窗口 · RSI-Opt 内外层 · 训练知识库 ─────────
  var TRAIN_ZH = { lr: "学习率", epochs: "最多轮数", hidden: "隐层宽度", rank: "交叉层秩", emb: "根因嵌入维度", l2: "L2", bias_l2: "偏置 L2", anchor: "锚定上一版",
    patience: "平台期轮数", hard_share: "困难题比例", decoy_share: "干扰题比例", receipt_weight: "产线样本权重", batch: "批大小" };
  var CTL_ZH = { lr_decay: "学习率减半", rollback: "回滚 + 减半", early_stop: "提前停止", halt: "停止" };
  function lossCurve(log) {
    var ep = (log && log.epochs) || [];
    if (!ep.length) return "";
    var W = 520, H = 150, P = 28, vals = ep.map(function (e) { return e.val; }).concat(ep.map(function (e) { return e.train; })), lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var x = function (i) { return P + i * (W - 2 * P) / Math.max(1, ep.length - 1); }, y = function (v) { return H - P + (lo - v) * (H - 2 * P) / Math.max(1e-6, hi - lo); };
    var line = function (k) { return ep.map(function (e, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(e[k]).toFixed(1); }).join(" "); };
    var marks = ((log.decisions) || []).map(function (d) { var i = Math.min(ep.length - 1, d.epoch - 1); return '<line x1="' + x(i) + '" x2="' + x(i) + '" y1="' + P / 2 + '" y2="' + (H - P) + '" stroke="var(--warn)" stroke-dasharray="3 3"/><text x="' + (i > ep.length * 0.75 ? x(i) - 3 : x(i) + 3) + '" y="' + (P / 2 + 8) + '" font-size="10" fill="var(--muted)"' + (i > ep.length * 0.75 ? ' text-anchor="end"' : "") + ">" + esc(CTL_ZH[d.action] || d.action) + "</text>"; }).join("");
    return '<svg viewBox="0 0 ' + W + " " + H + '" style="width:100%;max-width:560px"><path d="' + line("train") + '" fill="none" stroke="var(--muted)" stroke-width="1.5"/><path d="' + line("val") + '" fill="none" stroke="var(--accent)" stroke-width="2"/>' + marks +
      '<text x="' + P + '" y="' + (H - 6) + '" font-size="10" fill="var(--muted)">epoch 1 … ' + ep.length + "　灰：训练损失　蓝：验证损失　虚线：控制器决定</text></svg>";
  }
  function pageFastbrain(head) {
    return api("fastbrain").then(function (o) {
      var a = o.active, m = a ? a.metrics : {}, ar = a ? a.params : {};
      var tiles = [[a ? "v" + a.id : "未晋升", a ? "在用快脑模型（" + (ar.hidden || "-") + " 隐层 · 交叉层秩 " + (ar.rank || 0) + " · 嵌入 " + (ar.emb || 0) + " · " + ((a.window || {}).params || "-") + " 参数）" : "还没有通过晋升门的模型，快脑退回贝叶斯"],
        [a ? m.candidate.hit1 + " vs " + m.bayes.hit1 : "-", "留出集首位命中率：快脑模型 vs 贝叶斯"],
        [a && m.test ? m.test.model + " vs " + m.test.bayes : "-", "RSI 测试集（只用于报告）：快脑模型 vs 贝叶斯"],
        [a ? m.candidate.ece + " / " + m.bayes.ece : "-", "校准误差 ECE：模型 / 贝叶斯（越小越可信）"],
        [a && a.conformal ? pct(a.conformal.coverage_holdout) : "-", "保形集合在留出集上的覆盖（目标 90%）"]];
      view.innerHTML = head + '<div class="tiles">' + tiles.map(function (t) { return '<div class="tile"><div class="v">' + esc(t[0]) + '</div><div class="l">' + esc(t[1]) + "</div></div>"; }).join("") + "</div>" +
        '<div class="panel"><div class="flow"><div class="flow-src"><div class="flow-box ok"><b>训练什么</b><span>快脑小模型：分数 = 宽（线性，初始化为贝叶斯证据权重）+ 深（隐层）+ 交叉（DCN-v2）+ 根因偏置</span><span class="muted small">特征 = 贝叶斯 5 个证据因子 + 结构信号 + 原始信号（连续良率在变点后时间窗里逐机台检验）；未训练时 = 贝叶斯</span></div>' +
        '<div class="flow-box accent"><b>A-RSI 训练窗口（何时更新、最后是否更新）</b><span>有触发才开窗 → 课程 agent 按固定配额出题、孪生给标签 + 产线确认样本 → 从上一版继续训练</span><span>健康门（不过重试一次，再不过就停）→ 晋升门（留出集上不比在用版本和贝叶斯差）</span></div></div>' +
        '<div class="flow-arrow"><b class="arr">↔</b><span class="muted small">RSI-Opt</span></div>' +
        '<div class="flow-box mem"><b>RSI-Opt 内外两层</b><span>内层：每个 epoch 末控制器决定继续 / 减半 / 回滚 / 停止</span><span>外层：跨 run 进化训练策略（学习率、结构、正则、出题配比），select 集验收 + report 集确认</span><span class="muted small">训练知识库：事实（数据与环境）/ 做法（训练技巧），失败追责，停用留档</span></div></div></div>' +
        (a ? '<div class="panel" style="margin-top:16px"><h2>在用模型 v' + a.id + " 的训练过程</h2>" + lossCurve(a.log) +
          '<p class="small">' + ((a.log || {}).decisions || []).map(function (d) { return "第 " + d.epoch + " 轮：" + esc(d.why) + (d.helped == null ? "" : d.helped ? "（之后改善了）" : "（之后没改善）"); }).join("；") + "</p></div>" : "") +
        '<div class="panel" style="margin-top:16px"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">训练窗口（A-RSI）</h2><span class="muted small">待触发：' + (o.triggers.window.length ? o.triggers.window.map(esc).join("、") : "无") + '</span><span class="spacer"></span><button class="btn ghost sm" id="fb-train">▶ 手动开一个窗口</button></div>' +
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>时间</th><th>触发</th><th>健康门</th><th>控制器</th><th>样本池信息量</th><th>留出集：候选 / 在用 / 贝叶斯</th><th>对数损失差</th><th>结果</th></tr></thead><tbody>' +
        o.windows.map(function (x) { var me = x.metrics || {}, w = x.window || {}, d = me.diff_logloss; return "<tr><td class=\"num\">" + x.id + '</td><td class="num">' + d10(x.created_at) + '</td><td class="small">' + esc(w.trigger || "") + ((w.focus || []).length ? '<div class="muted">加倍：' + w.focus.map(symName).map(esc).join("、") + "</div>" : "") + '</td><td class="small">' + (w.health && w.health.ok ? "通过" : "未通过") + (w.attempts > 1 ? "（重试 " + (w.attempts - 1) + " 次）" : "") + '</td><td class="small">' + (((x.log || {}).decisions) || []).map(function (dd) { return esc(CTL_ZH[dd.action] || dd.action) + "@" + dd.epoch; }).join(" ") + '</td><td class="num">' + (w.informative == null ? "-" : pct(w.informative)) + (w.refresh_next ? ' <span class="badge warn">下次换池</span>' : "") + '</td><td class="num">' + (me.candidate ? me.candidate.hit1 + " / " + me.incumbent.hit1 + " / " + me.bayes.hit1 : "-") + '</td><td class="num">' + (d ? d.mean + " [" + d.ci95[0] + ", " + d.ci95[1] + "]" : "-") + "</td><td>" + '<span class="badge ' + ({ active: "ok", rejected: "warn", failed: "warn" }[x.status] || "") + '">' + esc({ active: "在用", rejected: "未晋升", failed: "健康门失败", retired: "已替换" }[x.status] || x.status) + '</span><div class="muted small">' + esc(me.gate_note || "") + "</div></td></tr>"; }).join("") + "</tbody></table></div>" +
        '<p class="muted small"><b>何时开窗</b>：没有窗口 / 新增 ≥20 条现场确认 / 孪生同步了产线变更 / 样本池饱和 / 慢脑推翻快脑 ≥8 次 / 产线命中率比留出集低 10 个百分点。<b>晋升门</b>：对数损失配对区间上界 ≤ 0、命中率下界 ≥ −2 个百分点、ECE 不变差、对贝叶斯命中率下界 ≥ −2 个百分点。</p></div>' +
        '<div class="grid g2" style="margin-top:16px"><div class="panel"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">训练策略存档（RSI-Opt 外层）</h2><span class="spacer"></span><button class="btn ghost sm" id="fb-opt">▶ 手动跑一次</button></div>' +
        '<p class="small" style="margin-top:0">在用策略 ' + (o.policy.id ? "#" + o.policy.id : "默认") + "：" + Object.keys(o.policy.params).map(function (k) { return esc(TRAIN_ZH[k] || k) + " " + o.policy.params[k]; }).join(" · ") + '<br><span class="muted">待触发：' + (o.triggers.rsi_opt.length ? o.triggers.rsi_opt.map(esc).join("、") : "无（连续 2 个窗口失败才跑）") + "</span></p>" +
        '<div class="table-wrap"><table><thead><tr><th>#</th><th>改动</th><th>select 对数损失</th><th>结果</th></tr></thead><tbody>' +
        o.policies.slice(0, 12).map(function (p) { var s = (p.scores || {}).select; return "<tr><td class=\"num\">" + p.id + '</td><td class="small">' + ((p.edits || []).map(function (e) { return "<b>" + esc(TRAIN_ZH[e.component] || e.component) + "</b> " + e.from + " → " + e.to; }).join("<br>") || "基线") + '</td><td class="num">' + (s ? s.logloss : "-") + "</td><td>" + '<span class="badge ' + ({ active: "ok", accepted: "ok", rejected: "warn" }[p.status] || "") + '">' + esc({ active: "在用", accepted: "采纳", rejected: "拒绝", retired: "已替换" }[p.status] || p.status) + '</span><div class="muted small">' + esc(p.note || "") + "</div></td></tr>"; }).join("") + "</tbody></table></div></div>" +
        '<div class="panel"><h2>训练知识库（RSI-Opt 的跨 run 知识）</h2><div class="chips">' + [["fact", "事实"], ["tip", "做法"]].map(function (t) { return ["stable", "active", "candidate", "retired"].map(function (s) { var n = o.notes.counts[t[0] + ":" + s]; return n ? '<span class="chip">' + t[1] + "·" + STATUS_ZH[s] + " " + n + "</span>" : ""; }).join(""); }).join("") + "</div>" +
        '<ul class="timeline small" style="margin-top:8px;max-height:340px;overflow:auto">' + o.notes.items.slice(0, 18).map(function (x) { return '<li class="' + (x.status === "retired" ? "refute" : x.status === "stable" ? "confirm" : "") + '"><span class="badge">' + (x.track === "fact" ? "事实" : "做法") + "</span> " + esc(x.text) + ' <span class="muted">成功率 ' + pct(x.p) + " · " + x.n + " 次" + (x.retired_reason ? " · 停用：" + esc(x.retired_reason) : "") + "</span></li>"; }).join("") + "</ul></div></div>";
      var bind = function (id, path, msg) { var b = document.getElementById(id); if (b) b.onclick = function () { b.disabled = true; b.textContent = "运行中…"; post(path).then(function () { toast(msg); route(); }).catch(function (e) { toast(e.message); b.disabled = false; }); }; };
      bind("fb-train", "fastbrain/train", "训练窗口完成"); bind("fb-opt", "fastbrain/optimize", "训练策略进化完成");
    });
  }

  var EVO_TABS = [
    ["overview", "概览", "看自我进化有没有用：两种学习加起来让根因命中率提高了多少、证据权重学成了什么样、和不做 RSI 相比好多少。"],
    ["rsi", "RSI 学习", "智能体在数字孪生里自己出题、作答、验证：每次会话的得分曲线、每道练习题，以及每次权重更新是否通过验收门。"],
    ["continual", "持续学习", "从产线真实案例学习：工程师每确认一个根因，根因先验和机台热点就更新；学习曲线对比“持续学习”和“冻结在教科书先验”。"],
    ["fastbrain", "快脑训练", "快脑是一个训练出来的小模型（Wide & Deep & Cross 排序网络）：A-RSI 决定何时开训练窗口、最后是否晋升；RSI-Opt 内层在训练中做决定、外层跨 run 进化训练策略；训练知识库记录事实与做法。"],
    ["harness", "快慢双脑", "Harness 层：快脑模型把握够就直接给结论，否则慢脑让两个挑战者（近邻训练、贝叶斯意见）在近邻探测题上和快脑比；改进经验库记录哪个组件在什么条件下有效；RRSI 有触发才从改进库提议修改基础 harness。"]];
  function evoTabs(tab) {
    var cur = EVO_TABS.find(function (t) { return t[0] === tab; }) || EVO_TABS[0];
    return '<div class="subtabs">' + EVO_TABS.map(function (t) { return '<a href="#/evolution?tab=' + t[0] + '" class="' + (t[0] === cur[0] ? "on" : "") + '">' + t[1] + "</a>"; }).join("") + '</div><p class="tab-desc">' + cur[2] + "</p>";
  }

  function pageEvolution(params) {
    var q = new URLSearchParams(params || ""), tab = { twin: "rsi", line: "continual" }[q.get("tab")] || q.get("tab") || (q.get("s") ? "rsi" : "overview");
    if (["overview", "rsi", "continual", "fastbrain", "harness"].indexOf(tab) < 0) tab = "overview";
    var head = '<div class="page-head"><h1>自我进化</h1><span class="muted small">RSI 学习和持续学习是同一件事：把验证过的结论写进同一套记忆，区别只在标签来自数字孪生还是产线。</span><span class="spacer"></span>' +
      '<button class="btn" id="evo-run">▶ 启动自我进化会话</button></div>' + evoTabs(tab);
    var bindRun = function () { var b = document.getElementById("evo-run"); if (b) b.onclick = function () { sendChat("启动一次自我进化会话", "rsi"); }; };
    if (tab === "overview") return Promise.all([api("rsi/benchmark"), api("learning/weights"), api("kpis"), api("rsi/sessions")]).then(function (r) {
      var b = r[0] ? r[0].results : null, W = r[1], L = r[2].learning, bt = L.backtest || {}, s = (r[3][0] || {}).summary || {};
      var v2 = b && b.modes.v2, none = b && b.modes.none;
      view.innerHTML = head + '<div class="tiles">' +
        '<div class="tile"><div class="v" style="color:var(--accent)">' + (v2 ? v2.hit1_mean : "-") + '</div><div class="l">新产线冷启动 · v2 自我进化后（不做：' + (none ? none.hit1_mean : "-") + "）</div></div>" +
        '<div class="tile"><div class="v">' + hitPlain(s) + '</div><div class="l">最近一次 RSI 会话 · 留出集首位命中率</div></div>' +
        '<div class="tile"><div class="v">' + pct(bt.hit1_learning) + '</div><div class="l">真实案例持续学习（冻结先验 ' + pct(bt.hit1_frozen) + "）</div></div>" +
        '<div class="tile"><div class="v">' + (W.current ? "n=" + W.current.n : "-") + '</div><div class="l">证据权重：已验证样本 · ' + W.history.length + " 个版本</div></div>" +
        '<div class="tile"><div class="v">' + L.memory_active + '</div><div class="l">生效记忆（孪生验证 ' + L.memory_sandbox + "）</div></div></div>" +
        '<div class="panel">' + sourcesDiagram() + "</div>" + liveBox() +
        '<div class="grid g2" style="margin-top:16px"><div class="panel"><h2>证据权重：每类证据该信多少（v2）</h2>' + weightsCard(W) + "</div>" +
        '<div class="panel"><h2>新产线冷启动消融（如实报告）</h2>' + benchBars(b) + "</div></div>" +
        '<div class="panel" style="margin-top:16px"><h2>v2 完整会话的逐步得分（代表性种子）</h2>' + rsiCurve(b && b.modes.v2 ? b.modes.v2.curve : [], { h: 210 }) +
        '<p class="muted small">广度 BRS 12 题 → 证据校准 CAL 480 题（每波 24 题并行）→ 深度 DRS 最多 36 题（按信息价值选题）。曲线在校准阶段抬升最快：这是“学证据权重”而不只是“学先验”的效果。</p></div>';
      bindRun();
    });
    if (tab === "rsi") return Promise.all([api("rsi/sessions"), api("rsi/gaps")]).then(function (r) {
      var sessions = r[0], gaps = r[1], sid = +q.get("s") || (sessions[0] && sessions[0].id) || 0;
      return (sid ? api("rsi/sessions/" + sid) : Promise.resolve(null)).then(function (d) {
        var sess = d ? d.session : null, tasks = d ? d.tasks : [], sm = sess ? sess.summary || {} : {};
        view.innerHTML = head + '<div class="grid g3"><div class="panel"><h2>🎓 课程 agent</h2><p class="muted small">独占“练什么”。按<b>信息价值</b>选题：知识梯度 KG(*) × 证据模糊率，即“再练几道这类题，期望命中率能涨多少”；另出一批证据校准题。连续几题不再带来新知识就停。</p></div>' +
          '<div class="panel"><h2>🔧 执行 agent</h2><p class="muted small">唯一能写记忆的角色。用当前记忆做根因分析并提出整改；验证判定之后才合并经验：先验、热点、证据权重的训练样本、带条件的流程与教训。</p></div>' +
          '<div class="panel"><h2>⚖️ 验证 agent</h2><p class="muted small">独占“判真假”。只看孪生的干预反馈：按提议整改后同一时段良率是否恢复。看不到执行 agent 的推理和记忆；改善低于量测噪声一律不通过。</p></div></div>' + liveBox() +
          '<div class="panel" style="margin-top:16px"><h2>会话时间线 <span class="muted" style="font-weight:400;font-size:12px">产线变更同步到数字孪生后，下一次会话把变化学进来；点一行查看详情</span></h2>' + sessionTable(sessions, sid) + "</div>" +
          '<div class="panel" style="margin-top:16px"><h2>会话' + (sess ? " #" + sess.id + ' <span class="muted" style="font-weight:400">' + dt(sess.created_at) + " · " + esc(trigName(sess.trigger)) + " · " + esc({ done: "已完成", running: "进行中", error: "出错" }[sess.status] || sess.status) + "</span>" : "") + "</h2>" +
          (sess && (sess.config || {}).changes && sess.config.changes.length ? '<div class="callout">本次会话之前同步到孪生的产线变更：<b>' + changeText(sess.config) + "</b>。课程 agent 把这些症状的校准题比例提高到 4 倍，这部分校准题也会更新先验。</div>" : "") +
          (sess && sm.plateau ? '<div class="callout">命中率维持 <b>' + sm.hit1_after + "</b>（持平）：" + plateauWhy(sm) + "。</div>" : "") +
          (sess ? '<div class="big-compare"><span class="n f">' + (sm.hit1_before == null ? "-" : sm.hit1_before) + '</span><span>→</span><span class="n l">' + (sm.hit1_after == null ? "-" : sm.hit1_after) + '</span><span>留出集首位命中率</span><span class="muted">真因后验 ' + sm.partial_before + " → " + sm.partial_after + " · " + sm.tasks + " 题（广度 " + sm.brs + " · 校准 " + (sm.cal || 0) + " · 深度 " + sm.drs + "）· 权重重拟合 采纳 " + (sm.refits_accepted || 0) + " / 拒绝 " + (sm.refits_rejected || 0) + "</span></div>" + rsiCurve(sess.curve) +
            '<p class="muted small">在已有产线记忆之上练习：孪生经验权重 0.5；校准题只用来学证据权重，不改动产线先验（产线变更涉及的症状除外）。</p>' : '<div class="empty">尚无会话</div>') + "</div>" +
          (sess && sess.weights_log && sess.weights_log.length ? '<div class="panel" style="margin-top:16px"><h2>证据权重重拟合与验收门（本会话）</h2><div class="table-wrap"><table><thead><tr><th>阶段</th><th>样本</th><th>候选权重</th><th>面板命中率</th><th>对数损失</th><th>结果</th></tr></thead><tbody>' +
            sess.weights_log.map(function (w) { var g = w.gate || {}; return "<tr><td>" + esc(w.phase) + '</td><td class="num">' + w.n + '</td><td class="small">' + Object.keys(w.weights || {}).map(function (k) { return esc(k) + " ×" + w.weights[k].toFixed(2); }).join(" · ") + '</td><td class="num">' + (g.before ? g.before.hit1 + " → " + g.after.hit1 : "-") + '</td><td class="num">' + (g.before ? g.before.logloss + " → " + g.after.logloss : "-") + "</td><td>" + (w.accepted ? '<span class="badge ok">采纳</span>' : '<span class="badge warn" title="' + esc(w.reason) + '">拒绝</span>') + "</td></tr>"; }).join("") + "</tbody></table></div></div>" : "") +
          '<div class="panel" style="margin-top:16px"><h2>练习题明细（课程出题 → 执行作答 → 验证判定 → 记忆合并）</h2>' + practiceTable(tasks) + "</div>" +
          '<div class="panel" style="margin-top:16px"><h2>知识缺口 · 信息价值（课程 agent 的出题依据）</h2><div class="table-wrap"><table><thead><tr><th>症状</th><th>工段</th><th>不确定性（bit）</th><th>产线确认</th><th>孪生确认</th><th>知识梯度 KG(*)</th><th>期望增益（pp/题）</th></tr></thead><tbody>' +
          gaps.map(function (g) { return "<tr><td>" + esc(g.name) + "</td><td>" + esc(lbl("stages", g.stage)) + '</td><td class="num">' + g.entropy + '</td><td class="num">' + g.real + '</td><td class="num">' + g.sandbox + '</td><td class="num">' + g.kg + '</td><td class="num"><b>' + g.voi + "</b></td></tr>"; }).join("") + "</tbody></table></div></div>";
        bindRows(); bindRun();
      });
    });
    if (tab === "harness") return pageHarness(head).then(bindRun);
    if (tab === "fastbrain") return pageFastbrain(head).then(bindRun);
    if (tab === "continual") return Promise.all([api("learning/curve"), api("learning/edges"), api("learning/events?limit=60"), api("memory/hotspots")]).then(function (r) {
      var curve = r[0], edges = r[1], events = r[2], hs = r[3], s = curve.latest ? curve.latest.summary : {};
      var bySym = {}; edges.forEach(function (e) { (bySym[e.symptom] = bySym[e.symptom] || []).push(e); });
      view.innerHTML = head + '<div class="panel"><div class="page-head" style="margin-bottom:6px"><h2 style="margin:0">真实案例学习曲线（预序评估 · ' + (s.cases || 0) + ' 个工程师确认的案例）</h2><span class="spacer"></span><button class="btn ghost sm" id="bt-run">↻ 重新回放</button></div>' + learningChart(curve.latest ? curve.latest.points : []) +
        '<p class="muted small">按时间回放每个已关闭案例：先用当时的记忆预测、再学习；对照组停留在教科书先验。每次关案还会重拟合证据权重（经验收门）。</p></div>' +
        '<div class="grid g2" style="margin-top:16px"><div class="panel"><h2>根因先验：症状 → 根因份额</h2>' +
        Object.keys(bySym).sort(function (a, b) { var n = function (k) { return bySym[k].reduce(function (t, e) { return t + e.confirmed + e.confirmed_sandbox + e.refuted; }, 0); }; return n(b) - n(a); }).map(function (sym) { return "<h3>" + esc(symName(sym)) + "</h3>" + bySym[sym].map(function (e) {
          return '<div class="prior-post"><span title="' + esc(e.name) + '">' + esc(e.name.slice(0, 16)) + (e.learned ? ' <span class="badge accent">新</span>' : "") + '<br><span class="muted" style="font-size:11px">产线 ' + e.confirmed + " · 孪生 " + e.confirmed_sandbox + (e.refuted ? " · 否定 " + e.refuted : "") + '</span></span><span class="pp-track"><span class="prior" style="width:' + e.prior * 100 + '%"></span><span class="post" style="width:' + e.posterior * 100 + '%"></span></span><span class="num">' + pct(e.prior) + " → <b>" + pct(e.posterior) + "</b></span></div>"; }).join(""); }).join("") +
        '<p class="muted small">浅色 = 教科书先验；深色 = 当前份额。产线确认权重 1.0，孪生确认权重 0.5。</p></div>' +
        '<div class="grid" style="align-content:start"><div class="panel"><h2>机台热点</h2><div class="table-wrap"><table><thead><tr><th>机台</th><th>根因</th><th>产线</th><th>孪生</th></tr></thead><tbody>' +
        hs.slice(0, 14).map(function (h) { return "<tr><td><b>" + esc(h.tool) + "</b></td><td>" + esc(h.cause.slice(0, 20)) + '</td><td class="num">' + h.real + '</td><td class="num">' + h.sandbox + "</td></tr>"; }).join("") + "</tbody></table></div></div>" +
        '<div class="panel"><h2>学习时间线（最近 60 条）</h2><ul class="timeline">' + events.map(eventLine).join("") + "</ul></div></div></div>";
      document.getElementById("bt-run").onclick = function () { post("learning/backtest").then(function () { toast("已重新回放"); route(); }).catch(function (e) { toast(e.message); }); };
      bindRun();
    });
  }

  // ── 改进周期 / 审计 ────────────────────────────────────
  function pageCycles() {
    return api("improvement/cycles").then(function (cs) {
      view.innerHTML = '<div class="page-head"><h1>改进周期</h1><span class="badge">' + cs.length + ' 次</span><span class="muted small">DBOS 定时复盘并重拟合证据权重（演示历史为每月一次）；也可在对话里说「运行改进周期」</span><span class="spacer"></span><button class="btn" id="run-cycle">▶ 立即运行</button></div>' +
        (cs.map(function (c, i) { return '<div class="panel" style="margin-bottom:12px">' + (i === 0 ? md(c.summary_md) : "<details><summary><b>#" + c.id + "</b> " + dt(c.created_at) + " · " + esc(c.trigger) + "</summary>" + md(c.summary_md) + "</details>") + "</div>"; }).join("") || '<div class="panel empty">暂无</div>');
      document.getElementById("run-cycle").onclick = function (e) { e.target.disabled = true; post("improvement/run").then(function () { toast("完成"); route(); }).catch(function (err) { toast(err.message); e.target.disabled = false; }); };
    });
  }
  function pageAudit() {
    return api("audit?limit=1500").then(function (rows) {
      view.innerHTML = '<div class="page-head"><h1>审计日志</h1><span class="badge">' + rows.length + " 条</span>" + (rows.length ? '<span class="muted small">' + d10(rows[rows.length - 1].created_at) + " 至 " + d10(rows[0].created_at) + "</span>" : "") + '<span class="muted small">谁在什么时候做了什么：建案、分析、CAPA、关案批准、产线变更同步、证据权重更新、RSI 会话、改进周期</span></div><div class="panel table-wrap"><table><thead><tr><th>时间</th><th>操作者</th><th>动作</th><th>对象</th><th>详情</th></tr></thead><tbody>' +
        rows.map(function (r) { return '<tr><td class="muted num">' + dt(r.created_at) + "</td><td>" + esc(r.actor) + '</td><td><span class="badge">' + esc(r.action) + "</span></td><td>" + esc(r.target) + '</td><td class="muted small">' + esc(JSON.stringify(r.detail)) + "</td></tr>"; }).join("") + "</tbody></table></div>";
    });
  }

  // ── 图标（线条风格，跟随文字颜色） ───────────────────────
  var IC = {
    pen: '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    overview: '<path d="M3 20h18"/><path d="M6 16v-5M11 16V5M16 16v-8M21 16v-3"/>',
    cases: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    capa: '<circle cx="12" cy="12" r="9"/><path d="m8 12.5 2.8 2.8L16.5 9"/>',
    evolution: '<path d="M11 3.5l1.9 4.6 4.6 1.9-4.6 1.9L11 16.5l-1.9-4.6L4.5 10l4.6-1.9z"/><path d="M18.5 14.5l.9 2.1 2.1.9-2.1.9-.9 2.1-.9-2.1-2.1-.9 2.1-.9z"/>',
    cycles: '<path d="M20 11a8 8 0 0 0-14.6-4.5M4 13a8 8 0 0 0 14.6 4.5"/><path d="M4 3v4h4M20 21v-4h-4"/>',
    audit: '<path d="M7 3h7l5 5v13H7z"/><path d="M14 3v5h5M10 13h6M10 17h4"/>',
    panel: '<rect x="3" y="4" width="18" height="16" rx="2.5"/><path d="M9 4v16"/>',
    send: '<path d="M12 19V5M5.5 11.5 12 5l6.5 6.5"/>',
    x: '<path d="M18 6 6 18M6 6l12 12"/>'
  };
  function icon(n) { return '<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + IC[n] + "</svg>"; }
  var NAV = [["/overview", "overview", "质量总览"], ["/cases", "cases", "案例库"], ["/capa", "capa", "CAPA 看板"], ["/evolution", "evolution", "自我进化"], ["/cycles", "cycles", "改进周期"], ["/audit", "audit", "审计日志"]];

  // ── 对话（AG-UI 1.0）：多条对话保存在本机浏览器 ─────────────
  function rid(p) { return p + "-" + Math.random().toString(36).slice(2, 10); }
  var TKEY = "kaizen_threads_v1", CKEY = "kaizen_current";
  function loadThreads() {
    try { var t = JSON.parse(localStorage.getItem(TKEY) || "[]"); if (Array.isArray(t)) { t.forEach(function (th) { (th.messages || []).forEach(function (m) { m.done = true; }); }); return t; } } catch (e) {}
    return [];
  }
  function newThread(mode) { return { id: rid("t"), title: "", mode: mode || "rca", messages: [], pending: null, updated: Date.now() }; }
  var THREADS = loadThreads();
  var CHAT = (function () { var id = null; try { id = sessionStorage.getItem(CKEY); } catch (e) {} return THREADS.find(function (t) { return t.id === id; }) || newThread(); })();
  function saveThreads() {
    THREADS = THREADS.filter(function (t) { return t.messages.length; }).sort(function (a, b) { return b.updated - a.updated; }).slice(0, 40);
    for (var n = THREADS.length; n >= 0; n--) { try { localStorage.setItem(TKEY, JSON.stringify(THREADS.slice(0, n))); break; } catch (e) {} }
    try { sessionStorage.setItem(CKEY, CHAT.id); } catch (e) {}
  }
  var LIVE = null, STATE = { workspace: { focusCaseId: null, openPage: null } }, BUSY = false;
  var PROMPTS = [
    { t: "分析 KZ-0138", m: "rca", d: "国产 DUV 套刻超规：看学到的知识怎样改变首位根因" },
    { t: "分析 KZ-0133", m: "rca", d: "刻蚀层边缘环形失效：晶圆图 + 共性分析 + FDC" },
    { t: "确认 KZ-0133 根因是刻蚀腔聚焦环磨损", m: "rca", d: "人在回路：你批准后才关案，产线确认写入记忆" },
    { t: "启动一次自我进化会话", m: "rsi", d: "三个 agent 在数字孪生里练习：广度 → 证据校准 → 深度" },
    { t: "学习效果怎么样？", m: "coach", d: "用数字总结 RSI 学习和持续学习的效果" },
    { t: "Fab A 刻蚀层 CP 良率下降，失效集中在晶圆外圈，良率=88%", m: "rca", d: "描述一个新问题：自动建案并分析" }];
  var MODES = { rca: "根因分析", coach: "改进教练", rsi: "自我进化" };
  var PAGE_NAME = { "/overview": "质量总览", "/cases": "案例库", "/capa": "CAPA 看板", "/evolution": "自我进化", "/cycles": "改进周期", "/audit": "审计日志" };
  var app = document.getElementById("app"), chatPane = document.getElementById("chat-pane"), pagePane = document.getElementById("page-pane");
  var chatLog = document.getElementById("chat-log"), input = document.getElementById("chat-input"), modeSel = document.getElementById("chat-mode");

  function actCard(a) {
    var c = a.content || {}, body = "";
    var target = c.route ? (/^\/cases\/(\d+)/.exec(c.route) ? code(/^\/cases\/(\d+)/.exec(c.route)[1]) : PAGE_NAME[c.route.split("?")[0]] || "页面") : "";
    var open = c.route ? '<a class="btn sm' + (c.open ? "" : " ghost") + '" href="#' + esc(c.route) + '">打开' + esc(target) + " →</a>" : "";
    var hz = c.harness ? '<div class="mini-stats"><span><b>' + (c.harness.route === "fast" ? (c.harness.fast_kind === "model" ? "快脑模型" : "快脑（贝叶斯）") : "慢脑") + "</b>" + (c.harness.chosen ? "（采用" + esc(CHALLENGER[c.harness.chosen] || c.harness.chosen) + "）" : "") + "</span><span>校准概率 <b>" + pct(c.harness.reliability) + "</b></span><span>90% 把握集合 <b>" + c.harness.set_n + "</b> 个根因</span></div>" : "";
    if (a.type === "kaizen.case" && c.top) body = c.top.map(function (h, i) { return '<div class="mini-hyp"><span>' + (i + 1) + ". " + esc(h.name) + '</span><span class="bar"><i style="width:' + Math.max(2, h.posterior * 100) + '%"></i></span><b class="num">' + pct(h.posterior) + "</b></div>"; }).join("") + (c.changed ? '<div class="muted small">🧠 学到的知识改变了首位根因</div>' : "") + hz;
    else if (a.type === "kaizen.rsi") {
      body = '<div class="mini-stats"><span>阶段 <b>' + esc(c.phase || "-") + "</b></span><span>已练 <b>" + (c.done || 0) + "</b> 题 · 通过 " + (c.passes || 0) + "</span><span>首位命中率 <b>" + (c.hit1_before == null ? "-" : c.hit1_before + " → " + c.hit1) + "</b></span></div>";
      if (LIVE && c.route && c.route.indexOf("s=" + (LIVE.state || {}).session_id) > 0) body += '<div class="act-live">' + rsiCurve(LIVE.state.curve || [], { h: 150 }) + '<ul class="timeline small">' + LIVE.log.slice(-5).map(function (l) { return '<li class="' + (l[1] || "") + '">' + l[0] + "</li>"; }).join("") + "</ul></div>";
    }
    else if (c.rows) body = '<ul class="small">' + c.rows.map(function (r) { return "<li>" + esc(r) + "</li>"; }).join("") + "</ul>" + (c.stats ? '<div class="mini-stats">' + Object.keys(c.stats).map(function (k) { return "<span>" + esc(k) + " <b>" + esc(c.stats[k]) + "</b></span>"; }).join("") + "</div>" : "");
    return '<div class="act-card"><div class="act-h"><span class="badge accent">' + esc({ "kaizen.case": "案例", "kaizen.learning": "持续学习", "kaizen.rsi": "RSI 学习", "kaizen.cycle": "改进周期", "kaizen.overview": "页面" }[a.type] || "页面") + "</span><b>" + esc(c.title || "") + '</b><span class="spacer"></span>' + open + "</div>" + body + "</div>";
  }
  function interruptCard(m) {
    var it = m.interrupt;
    if (m.resolved) return '<div class="callout ' + (m.resolved === "resolved" ? "ok" : "") + '">' + (m.resolved === "resolved" ? "✅ 已批准" : "已取消") + "：" + esc(it.message) + "</div>";
    return '<div class="interrupt"><b>⏸ 需要你的批准</b><p>' + esc(it.message) + '</p><input id="int-note" placeholder="备注（可选），如：QE 已现场确认"><div class="int-btns"><button class="btn" data-int="resolved">批准并关案</button><button class="btn ghost" data-int="cancelled">取消</button></div><div class="muted small">AG-UI 1.0 中断：这次运行以 outcome=interrupt 结束；批准后带 resume 发起下一次运行，才会真正写入记忆。</div></div>';
  }
  function welcome() {
    return '<div class="welcome"><div class="logo big">K</div><h1>有什么质量问题需要分析？</h1><div class="attention" id="attention"></div></div>';
  }
  function suggestions() {
    return '<div class="prompt-grid">' + PROMPTS.map(function (p, i) { return '<button class="prompt" type="button" data-pi="' + i + '"><span class="badge">' + MODES[p.m] + "</span><b>" + esc(p.t) + '</b><span class="muted small">' + esc(p.d) + "</span></button>"; }).join("") + "</div>";
  }
  var renderQueued = false;
  function renderChat() { if (renderQueued) return; renderQueued = true; requestAnimationFrame(function () { renderQueued = false; renderChatNow(); }); }
  function renderChatNow() {
    var empty = !CHAT.messages.length, sug = document.getElementById("chat-suggest");
    chatPane.classList.toggle("empty", empty);
    if (empty) {
      chatLog.innerHTML = '<div class="chat-inner">' + welcome() + "</div>"; sug.innerHTML = suggestions(); sug.hidden = false;
      sug.querySelectorAll("[data-pi]").forEach(function (b) { b.onclick = function () { var p = PROMPTS[+b.dataset.pi]; sendChat(p.t, p.m); }; });
      loadAttention(); updateComposer(); return;
    }
    sug.hidden = true; sug.innerHTML = "";
    var atBottom = chatLog.scrollHeight - chatLog.scrollTop - chatLog.clientHeight < 80;
    chatLog.innerHTML = '<div class="chat-inner">' + CHAT.messages.map(function (m) {
      if (m.role === "user") return '<div class="msg user"><div class="bubble">' + esc(m.content) + "</div></div>";
      return '<div class="msg bot"><div class="avatar">K</div><div class="msg-body">' + (m.reasoning ? "<details" + (m.done ? "" : " open") + '><summary class="muted small">思考过程（工作流步骤）</summary><pre class="steps-trace">' + esc(m.reasoning) + "</pre></details>" : "") +
        (m.content ? md(m.content) : m.done ? "" : '<div class="typing"><i></i><i></i><i></i></div>') + (m.activities || []).map(actCard).join("") +
        (m.interrupt ? interruptCard(m) : "") + (m.error ? '<div class="callout warn">' + esc(m.error) + "</div>" : "") + "</div></div>";
    }).join("") + "</div>";
    chatLog.querySelectorAll("[data-int]").forEach(function (b) { b.onclick = function () { resolveInterrupt(b.dataset.int); }; });
    if (atBottom || BUSY) chatLog.scrollTop = chatLog.scrollHeight;
    updateComposer();
  }
  function updateComposer() {
    var pend = !!CHAT.pending;
    document.getElementById("chat-form").classList.toggle("blocked", pend);
    document.getElementById("chat-send").classList.toggle("busy", BUSY);
    input.placeholder = pend ? "请先在上方批准或取消待处理的关案请求" : CHAT.messages.length ? "继续追问，或输入「确认 KZ-0133 根因是…」" : "描述现场问题，或输入「分析 KZ-0133」";
    modeSel.value = CHAT.mode || "rca";
  }
  function loadAttention() {
    Promise.all([api("kpis"), api("rsi/sessions")]).then(function (r) {
      var k = r[0].kpis, s = (r[1][0] || {}).summary || {}, el = document.getElementById("attention"); if (!el) return;
      el.innerHTML = '<a class="att" href="#/cases?status=open">📂 进行中案例 <b>' + k.cases_open + "</b></a>" + '<a class="att' + (k.capa_overdue ? " warn" : "") + '" href="#/capa?overdue=1">⏰ 逾期 CAPA <b>' + k.capa_overdue + "</b></a>" +
        (s.hit1_after != null ? '<a class="att ok" href="#/evolution?tab=rsi">🧬 最近一次 RSI 学习 <b>' + hitPlain(s) + "</b></a>" : "");
    }).catch(function () {});
  }

  function chatHistory() {
    return CHAT.messages.filter(function (m) { return m.content && (m.role === "user" || m.role === "assistant"); }).map(function (m, i) { return { id: "m" + i, role: m.role, content: m.content }; });
  }
  function runAgui(thread, reply, body) {
    BUSY = true; renderChat();
    var texts = {};
    return agui(Object.assign({ threadId: thread.id, runId: rid("run"), state: STATE, tools: [], context: [] }, body), function (ev) {
      if (ev.type === "REASONING_MESSAGE_CONTENT") reply.reasoning += ev.delta;
      else if (ev.type === "TEXT_MESSAGE_START") { if (reply.content && !texts[ev.messageId]) reply.content += "\n\n"; texts[ev.messageId] = 1; }
      else if (ev.type === "TEXT_MESSAGE_CONTENT") reply.content += ev.delta;
      else if (ev.type === "ACTIVITY_SNAPSHOT") reply.activities.push({ id: ev.messageId, type: ev.activityType, content: ev.content });  // 只在消息里放卡片：点开才在主区域完整显示
      else if (ev.type === "ACTIVITY_DELTA") { var a = reply.activities.find(function (x) { return x.id === ev.messageId; }); if (a) applyPatch(a.content, ev.patch); }
      else if (ev.type === "STATE_SNAPSHOT") { if (ev.snapshot.workspace) Object.assign(STATE.workspace, ev.snapshot.workspace); if (ev.snapshot.session_id) { LIVE = { state: ev.snapshot, log: [["会话 #" + ev.snapshot.session_id + " 已启动 · DBOS 工作流 " + esc(ev.snapshot.workflow_id)]], done: false }; renderLive(); } }
      else if (ev.type === "STATE_DELTA" && LIVE) { applyPatch(LIVE.state, ev.delta); renderLive(); }
      else if (ev.type === "STEP_STARTED" && LIVE) { LIVE.log.push(["<b>▶ " + esc(ev.stepName) + "</b>", "edge_added"]); renderLive(); }
      else if (ev.type === "CUSTOM" && LIVE) {
        var v = ev.value;
        if (ev.name === "rsi.task") LIVE.log.push(['<span class="badge ' + (v.verdict === "PASS" ? "ok" : "warn") + '">' + v.verdict + "</span> 练习 #" + v.id + " " + esc(v.symptom_name) + ' <span class="badge">' + esc(v.variant) + "</span> 作答 " + v.attempts + " 次" + (v.new_items ? " · 新记忆 " + v.new_items : ""), v.verdict === "PASS" ? "confirm" : "refute"]);
        else if (ev.name === "rsi.cal") LIVE.log.push(["证据校准一波 " + v.n + " 题，通过 " + v.passes + "（累计 " + v.done + " 题）", "confirm"]);
        else if (ev.name === "rsi.weights") LIVE.log.push(['<span class="badge ' + (v.accepted ? "ok" : "warn") + '">' + (v.accepted ? "采纳" : "拒绝") + "</span> 证据权重重拟合 n=" + v.n + (v.gate && v.gate.before ? " · 面板 " + v.gate.before.hit1 + " → " + v.gate.after.hit1 : "") + (v.accepted ? "" : " · " + esc(v.reason))]);
        else if (ev.name === "rsi.phase" && v.phase === "STOP") LIVE.log.push([esc(v.detail)]);
        renderLive();
      }
      else if (ev.type === "RUN_ERROR") reply.error = ev.message;
      else if (ev.type === "RUN_FINISHED") {
        reply.done = true;
        if (ev.outcome && ev.outcome.type === "interrupt") { reply.interrupt = ev.outcome.interrupts[0]; thread.pending = { interrupt: reply.interrupt, index: thread.messages.indexOf(reply) }; }
        if (LIVE && !LIVE.done && body.forwardedProps.mode === "rsi") { LIVE.done = true; if (currentPath() === "/evolution") setTimeout(route, 400); }
      }
      if (thread === CHAT) renderChat();
    }).catch(function (e) { reply.error = e.message; }).then(function () { reply.done = true; BUSY = false; thread.updated = Date.now(); saveThreads(); renderThreads(); renderChat(); });
  }
  function adopt(thread, text) {  // 第一条消息时把对话加入“最近对话”
    if (!thread.title) thread.title = text.replace(/\s+/g, " ").slice(0, 30);
    thread.updated = Date.now();
    if (THREADS.indexOf(thread) < 0) THREADS.unshift(thread);
    if (currentPath() === "/" || currentPath() === "/c/" + thread.id) { if (location.hash !== "#/c/" + thread.id) history.replaceState(null, "", "#/c/" + thread.id); }
    setTitle(); renderThreads();
  }
  function sendChat(text, mode) {
    text = String(text || "").trim(); if (!text) return;
    if (BUSY) { toast("上一条消息还在处理"); return; }
    if (CHAT.pending) { toast("请先处理待批准的关案请求"); return; }
    mode = mode || modeSel.value || CHAT.mode; if (mode !== "rsi") CHAT.mode = mode;  // 自我进化是一次性动作，发完回到原模式
    CHAT.messages.push({ role: "user", content: text });
    var reply = { role: "assistant", content: "", reasoning: "", activities: [], done: false }; CHAT.messages.push(reply);
    adopt(CHAT, text);
    var hist = chatHistory(); renderChat();
    runAgui(CHAT, reply, { messages: hist, forwardedProps: { mode: mode } });
  }
  function resolveInterrupt(status) {
    var th = CHAT, p = th.pending; if (!p || BUSY) return;
    var note = (document.getElementById("int-note") || {}).value || "", owner = th.messages[p.index];
    owner.resolved = status; th.pending = null;
    th.messages.push({ role: "user", content: status === "resolved" ? "✅ 批准" + (note ? "：" + note : "") : "取消关案" });
    var reply = { role: "assistant", content: "", reasoning: "", activities: [], done: false }; th.messages.push(reply);
    runAgui(th, reply, { messages: chatHistory(), forwardedProps: { mode: "rca" },
      resume: [{ interruptId: p.interrupt.id, status: status, payload: status === "resolved" ? { approved: true, reason: note } : null }] });
  }

  // ── 左侧栏：新对话 · 功能 · 最近对话 ───────────────────────
  function renderNav() {
    document.getElementById("chat-new").innerHTML = icon("pen") + "<span>新对话</span>";
    document.getElementById("sb-nav").innerHTML = NAV.map(function (n) { return '<a class="sb-item" href="#' + n[0] + '" data-path="' + n[0] + '">' + icon(n[1]) + "<span>" + n[2] + "</span></a>"; }).join("");
    document.getElementById("sb-collapse").innerHTML = icon("panel");
    document.getElementById("sb-open").innerHTML = icon("panel");
    document.getElementById("chat-send").innerHTML = icon("send");
  }
  function renderThreads() {
    var el = document.getElementById("sb-threads"), onChat = !currentPage();
    var list = THREADS.filter(function (t) { return t.messages.length; });
    el.innerHTML = list.length ? list.map(function (t) {
      return '<div class="sb-thread' + (t === CHAT && onChat ? " on" : "") + '"><a href="#/c/' + esc(t.id) + '" title="' + esc(t.title) + '">' + (t.pending ? '<i class="dot" title="有待批准的关案"></i>' : "") + "<span>" + esc(t.title || "新对话") + '</span></a><button type="button" data-del="' + esc(t.id) + '" title="删除对话" aria-label="删除对话">' + icon("x") + "</button></div>"; }).join("")
      : '<div class="sb-empty">还没有对话</div>';
    el.querySelectorAll("[data-del]").forEach(function (b) { b.onclick = function () {
      var t = THREADS.find(function (x) { return x.id === b.dataset.del; }); if (!t || (BUSY && t === CHAT)) return;
      if (!window.confirm("删除对话「" + (t.title || "新对话") + "」？只删除本机浏览器里的记录，案例和记忆不受影响。")) return;
      THREADS = THREADS.filter(function (x) { return x !== t; });
      if (t === CHAT) { CHAT = newThread(t.mode); if (!currentPage()) history.replaceState(null, "", "#/"); renderChat(); setTitle(); }
      saveThreads(); renderThreads(); }; });
  }
  function currentPath() { return (location.hash.replace(/^#/, "") || "/").split("?")[0]; }
  function currentPage() { var p = currentPath(); return p === "/" || p.indexOf("/c/") === 0 ? null : p; }
  function setTitle() { document.getElementById("main-title").textContent = currentPage() ? "" : CHAT.title || ""; }
  function closeDrawer() { app.classList.remove("sb-show"); }
  function isMobile() { return window.matchMedia("(max-width: 760px)").matches; }

  // ── 路由：默认显示对话；点左侧功能后在同一主区域完整显示 ─────
  function route() {
    var h = location.hash.replace(/^#/, "") || "/", parts = h.split("?"), path = parts[0], params = parts[1] || "", m, p;
    var legacy = { "/learning": "/evolution?tab=continual", "/rsi": "/evolution?tab=rsi", "/chat": "/" };
    if (legacy[path]) { location.replace("#" + legacy[path] + (path === "/rsi" && params ? "&" + params : "")); return; }
    closeDrawer();
    document.querySelectorAll("#sb-nav [data-path]").forEach(function (a) { var q = a.dataset.path; a.classList.toggle("on", path === q || path.indexOf(q + "/") === 0 || (q === "/cases" && /^\/cases\//.test(path))); });
    if (path === "/" || (m = /^\/c\/([\w-]+)$/.exec(path))) {
      if (m) { var t = THREADS.find(function (x) { return x.id === m[1]; }); if (t && t !== CHAT) { CHAT = t; try { sessionStorage.setItem(CKEY, CHAT.id); } catch (e) {} } else if (!t) { history.replaceState(null, "", "#/"); } }
      STATE.workspace.openPage = null;
      pagePane.hidden = true; chatPane.hidden = false; document.getElementById("back-chat").hidden = true; app.classList.remove("on-page");
      setTitle(); renderThreads(); renderChat(); if (!isMobile()) input.focus();
      return;
    }
    STATE.workspace.openPage = path;
    if ((m = /^\/cases\/(\d+)$/.exec(path))) STATE.workspace.focusCaseId = +m[1];
    chatPane.hidden = true; pagePane.hidden = false; document.getElementById("back-chat").hidden = false; app.classList.add("on-page");
    setTitle(); renderThreads(); pagePane.scrollTop = 0;
    if ((m = /^\/cases\/(\d+)$/.exec(path))) p = pageCase(+m[1], params);
    else if (path === "/overview") p = pageOverview();
    else if (path === "/cases") p = pageCases(params);
    else if (path === "/capa") p = pageCapa(params);
    else if (path === "/evolution") p = pageEvolution(params);
    else if (path === "/cycles") p = pageCycles();
    else if (path === "/audit") p = pageAudit();
    else { view.innerHTML = '<div class="empty">页面不存在</div>'; return; }
    p.catch(function (e) { view.innerHTML = '<div class="panel empty">加载失败：' + esc(e.message) + "</div>"; });
  }

  renderNav();
  try { if (localStorage.getItem("kaizen_sb_collapsed") === "1") app.classList.add("sb-collapsed"); } catch (e) {}
  document.getElementById("sb-collapse").onclick = function () { if (isMobile()) { closeDrawer(); return; } app.classList.add("sb-collapsed"); try { localStorage.setItem("kaizen_sb_collapsed", "1"); } catch (e) {} };
  document.getElementById("sb-open").onclick = function () { if (isMobile()) { app.classList.add("sb-show"); return; } app.classList.remove("sb-collapsed"); try { localStorage.setItem("kaizen_sb_collapsed", "0"); } catch (e) {} };
  document.getElementById("scrim").onclick = closeDrawer;
  document.getElementById("back-chat").onclick = function () { location.hash = CHAT.messages.length ? "#/c/" + CHAT.id : "#/"; };
  document.getElementById("chat-new").onclick = function () { if (CHAT.messages.length) CHAT = newThread(CHAT.mode); if (location.hash === "#/" || location.hash === "") route(); else location.hash = "#/"; };
  document.getElementById("chat-form").onsubmit = function (e) { e.preventDefault(); var t = input.value; if (CHAT.pending) { toast("请先处理待批准的关案请求"); return; } if (!t.trim() || BUSY) return; input.value = ""; input.style.height = ""; sendChat(t, modeSel.value); };
  input.onkeydown = function (e) { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); document.getElementById("chat-form").requestSubmit(); } };
  input.oninput = function () { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 200) + "px"; };
  modeSel.onchange = function () { if (modeSel.value !== "rsi") CHAT.mode = modeSel.value; };
  chatLog.addEventListener("click", function (e) { var a = e.target.closest("a"); if (!a) return; var href = a.getAttribute("href") || "", i = href.indexOf("#/"); if (i > 0 && href.indexOf(location.origin) === 0) { e.preventDefault(); location.hash = href.slice(i); } });

  // ── 首次启动：演示数据在后台生成，显示进度，生成完自动刷新 ──
  function demoBanner(d) {
    var el = document.getElementById("demo-banner"), busy = d && (d.state === "seeding" || d.state === "idle");
    if (!d || (!busy && d.state !== "error")) { el.hidden = true; return; }
    var pc = Math.round((d.progress || 0) * 100);
    el.hidden = false;
    el.innerHTML = d.state === "error" ? "<b>演示数据准备失败</b>：" + esc(d.detail) :
      '<span class="spin" aria-hidden="true"></span><b>正在生成演示数据 ' + pc + "%</b><span>" + esc(d.detail || "按时间回放去年至今的产线") + '</span><span class="muted">可以先浏览，数据会陆续出现</span><i style="width:' + pc + '%"></i>';
  }
  function watchDemo() {
    api("meta").then(function (m) {
      META.demo = m.demo; demoBanner(m.demo);
      if (m.demo && (m.demo.state === "seeding" || m.demo.state === "idle")) setTimeout(watchDemo, 2000); else route();
    }).catch(function () { setTimeout(watchDemo, 4000); });
  }

  api("meta").then(function (m) {
    META = m;
    document.title = "Kaizen";
    window.addEventListener("hashchange", route);
    route();
    demoBanner(m.demo);
    if (m.demo && (m.demo.state === "seeding" || m.demo.state === "idle")) setTimeout(watchDemo, 2000);
  }).catch(function (e) { chatLog.innerHTML = '<div class="panel empty">无法连接后端：' + esc(e.message) + "</div>"; });
})();
