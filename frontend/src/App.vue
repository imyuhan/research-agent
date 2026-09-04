<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from "vue";
import { ArrowRight, BadgeCheck, FileText, Play, RefreshCw, Search, Square, Sparkles } from "lucide-vue-next";
import {
  cancelResearch,
  checkHealth,
  createResearch,
  getResearch,
  getResearchEvents,
  getResearchReport,
  subscribeResearchEvents
} from "./lib/api";
import { renderMarkdown } from "./lib/markdown";

const RECENT_KEY = "research-agent:recent-jobs";
const SAMPLE_TOPICS = [
  "2026年8月6日 苹果公司发布了什么新产品",
  "请比较大模型 RAG 与 Agent 的区别",
  "2026 年 AI 搜索产品的主要趋势"
];
const STAGE_ORDER = ["planner", "researcher", "writer", "reviewer"];
const TABS = [
  { id: "report", label: "报告" },
  { id: "evidence", label: "证据" },
  { id: "raw", label: "原始数据" }
];

const topic = ref("");
const apiStatus = ref("checking");
const isSubmitting = ref(false);
const isStopping = ref(false);
const activeTab = ref("report");
const autoStream = ref(true);
const currentJob = ref(null);
const currentReport = ref(null);
const recentJobs = ref(loadRecentJobs());
const events = ref([]);
const eventSource = ref(null);

const formState = reactive({
  quickNote: "实时任务流"
});

const isTerminal = computed(() => {
  const status = currentJob.value?.status || "";
  return ["completed", "failed", "cancelled"].includes(status);
});

const totalTokens = computed(() => {
  const usage = currentJob.value?.token_usage || {};
  return Object.values(usage).reduce((sum, bucket) => sum + (bucket?.total_tokens || 0), 0);
});

const reportHtml = computed(() => renderMarkdown(currentReport.value?.draft || currentJob.value?.draft || ""));
const researchPlan = computed(() => currentJob.value?.plan || currentReport.value?.plan || []);
const reviewFeedback = computed(() => currentJob.value?.review_feedback || currentReport.value?.review_feedback || "");
const canCancelCurrentJob = computed(() => {
  const status = currentJob.value?.status || "";
  return ["queued", "running"].includes(status);
});
const primaryActionLabel = computed(() => {
  if (currentJob.value?.status === "cancelling") return "取消中";
  if (isStopping.value) return "停止中";
  if (canCancelCurrentJob.value) return "停止研究";
  if (isSubmitting.value) return "启动中";
  return "开始研究";
});
const primaryActionIcon = computed(() => (canCancelCurrentJob.value || currentJob.value?.status === "cancelling" ? Square : Play));
const primaryActionTone = computed(() => (canCancelCurrentJob.value || currentJob.value?.status === "cancelling" ? "danger" : "accent"));
const primaryActionDisabled = computed(
  () =>
    isSubmitting.value ||
    isStopping.value ||
    currentJob.value?.status === "cancelling" ||
    (!canCancelCurrentJob.value && !topic.value.trim())
);

const currentTokenRows = computed(() => {
  const usage = currentJob.value?.token_usage || {};
  return Object.entries(usage).map(([node, bucket]) => ({
    node,
    prompt: bucket?.prompt_tokens || 0,
    completion: bucket?.completion_tokens || 0,
    total: bucket?.total_tokens || 0
  }));
});

const stageCards = computed(() => {
  const job = currentJob.value;
  return [
    {
      key: "planner",
      label: "规划",
      icon: Sparkles,
      state: stageState("planner"),
      value: `${job?.plan?.length || 0}`,
      hint: "子问题"
    },
    {
      key: "researcher",
      label: "研究",
      icon: Search,
      state: stageState("researcher"),
      value: `${job?.findings?.length || 0} 条结论`,
      hint: `${job?.evidence_pack?.length || 0} 条证据`
    },
    {
      key: "writer",
      label: "写作",
      icon: FileText,
      state: stageState("writer"),
      value: `${Math.max(0, job?.draft?.length || 0)} 字`,
      hint: "报告草稿"
    },
    {
      key: "reviewer",
      label: "审核",
      icon: BadgeCheck,
      state: stageState("reviewer"),
      value: `${job?.review_score || 0}/10`,
      hint: job?.review_passed ? "已通过" : "等待审核"
    }
  ];
});

const recentJobCards = computed(() => recentJobs.value.slice(0, 6));

function loadRecentJobs() {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveRecentJobs() {
  localStorage.setItem(RECENT_KEY, JSON.stringify(recentJobs.value.slice(0, 12)));
}

function upsertRecentJob(job) {
  if (!job) return;
  const item = {
    job_id: job.job_id,
    topic: job.topic,
    status: job.status,
    review_score: job.review_score || 0,
    updated_at: new Date().toISOString()
  };
  recentJobs.value = [item, ...recentJobs.value.filter((entry) => entry.job_id !== job.job_id)].slice(0, 12);
  saveRecentJobs();
}

function stageState(key) {
  const job = currentJob.value;
  if (!job) return "idle";
  const index = STAGE_ORDER.indexOf(key);
  const currentIndex = STAGE_ORDER.indexOf(job.current_step || "init");

  if (job.status === "failed" && key === job.current_step) return "error";
  if (job.status === "cancelled") return index <= currentIndex ? "stalled" : "idle";
  if (job.status === "completed") return "done";
  if (index < currentIndex) return "done";
  if (index === currentIndex) return "active";
  return "idle";
}

function statusLabel(status) {
  const map = {
    queued: "排队中",
    running: "运行中",
    cancelling: "取消中",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消"
  };
  return map[status] || "空闲";
}

function stageLabel(state) {
  const map = {
    idle: "等待",
    active: "进行中",
    done: "完成",
    error: "失败",
    stalled: "停止"
  };
  return map[state] || "等待";
}

function stepLabel(step) {
  const map = {
    init: "初始化",
    planner: "规划",
    researcher: "研究",
    writer: "写作",
    reviewer: "审核",
    done: "完成",
    unknown: "未知"
  };
  return map[step] || step || "未开始";
}

function tokenNodeLabel(node) {
  const map = {
    planner: "规划",
    researcher: "研究",
    writer: "写作",
    reviewer: "审核"
  };
  return map[node] || node;
}

function formatTime(value) {
  if (!value) return "—";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

function formatEventType(type) {
  const map = {
    "job.created": "任务创建",
    "job.started": "任务开始",
    "job.completed": "任务完成",
    "job.failed": "任务失败",
    "job.cancelled": "任务取消",
    "job.cancelling": "正在取消",
    "planner.completed": "规划完成",
    "researcher.completed": "研究完成",
    "writer.completed": "写作完成",
    "reviewer.completed": "审核完成"
  };
  return map[type] || type.replaceAll(".", " · ");
}

function formatEventMessage(event) {
  const map = {
    "job.created": "任务已创建",
    "job.started": "研究任务已开始",
    "job.completed": "研究任务已完成",
    "job.failed": "研究任务失败",
    "job.cancelled": "研究任务已取消",
    "job.cancelling": "正在取消任务",
    "planner.completed": "规划阶段完成",
    "researcher.completed": "研究阶段完成",
    "writer.completed": "写作阶段完成",
    "reviewer.completed": "审核阶段完成"
  };
  return map[event.type] || event.message || event.type;
}

function closeStream() {
  if (eventSource.value) {
    eventSource.value.close();
    eventSource.value = null;
  }
}

async function hydrateJob(jobId) {
  const [job, history, report] = await Promise.all([
    getResearch(jobId),
    getResearchEvents(jobId),
    getResearchReport(jobId)
  ]);

  currentJob.value = job;
  currentReport.value = report;
  events.value = history.events || [];
  upsertRecentJob(job);

  if (autoStream.value && !isTerminal.value) {
    openStream(jobId, events.value.length);
  } else {
    closeStream();
  }
}

function openStream(jobId, after = 0) {
  closeStream();
  eventSource.value = subscribeResearchEvents(jobId, after, {
    onEvent: async (event) => {
      events.value = [...events.value, event];
      if (event.type === "job.completed" || event.type === "job.failed" || event.type === "job.cancelled") {
        await hydrateJob(jobId);
      } else if (event.type.endsWith(".completed")) {
        const job = await getResearch(jobId);
        currentJob.value = job;
        upsertRecentJob(job);
      }
    },
    onError: () => {}
  });
}

async function startResearch() {
  const value = topic.value.trim();
  if (!value || isSubmitting.value) return;
  isSubmitting.value = true;
  try {
    const job = await createResearch(value);
    await hydrateJob(job.job_id);
  } catch (error) {
    console.error(error);
    apiStatus.value = "offline";
  } finally {
    isSubmitting.value = false;
  }
}

async function handlePrimaryAction() {
  if (canCancelCurrentJob.value) {
    await stopResearch();
    return;
  }
  await startResearch();
}

async function openRecent(jobId) {
  try {
    await hydrateJob(jobId);
    activeTab.value = "report";
  } catch (error) {
    console.error(error);
  }
}

async function stopResearch() {
  const jobId = currentJob.value?.job_id;
  if (!jobId || isTerminal.value) return;
  isStopping.value = true;
  try {
    await cancelResearch(jobId);
    await hydrateJob(jobId);
  } finally {
    isStopping.value = false;
  }
}

async function refreshCurrent() {
  const jobId = currentJob.value?.job_id;
  if (!jobId) return;
  await hydrateJob(jobId);
}

function fillSample(sample) {
  topic.value = sample;
}

function refreshApiStatus() {
  apiStatus.value = "checking";
  checkHealth()
    .then(() => {
      apiStatus.value = "online";
    })
    .catch(() => {
      apiStatus.value = "offline";
    });
}

watch(autoStream, (value) => {
  if (!value) {
    closeStream();
  } else if (currentJob.value && !isTerminal.value) {
    openStream(currentJob.value.job_id, events.value.length);
  }
});

watch(
  () => currentJob.value?.status,
  (status) => {
    if (status && ["completed", "failed", "cancelled"].includes(status)) {
      closeStream();
    }
  }
);

onMounted(async () => {
  refreshApiStatus();
  if (recentJobs.value.length) {
    await openRecent(recentJobs.value[0].job_id);
  }
});

onBeforeUnmount(() => {
  closeStream();
});
</script>

<template>
  <div class="app-shell">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">RA</div>
        <div>
          <p class="eyebrow">研究工作台</p>
          <h1>智能研究助手</h1>
        </div>
      </div>

      <div class="topbar-actions">
        <span class="status-chip" :data-tone="apiStatus">
          {{ apiStatus === "online" ? "后端在线" : apiStatus === "offline" ? "后端离线" : "检查中" }}
        </span>
        <button class="icon-button" type="button" @click="refreshCurrent" :disabled="!currentJob">
          <RefreshCw :size="16" />
          刷新
        </button>
      </div>
    </header>

    <main class="workspace">
      <section class="panel composer-panel">
        <div class="panel-head">
          <div>
            <p class="section-kicker">新任务</p>
            <h2>发起研究</h2>
          </div>
          <span class="panel-note">{{ formState.quickNote }}</span>
        </div>

        <form class="composer-form" @submit.prevent="handlePrimaryAction">
          <label class="field">
            <span>研究主题</span>
            <textarea
              v-model="topic"
              rows="7"
              placeholder="输入研究主题，例如：2026年8月6日 苹果公司发布了什么新产品"
            />
          </label>

          <div class="toggle-row">
            <label class="toggle">
              <input v-model="autoStream" type="checkbox" />
              <span>实时更新</span>
            </label>
            <span class="toggle-hint">通过 SSE 接收进度</span>
          </div>

          <div class="sample-row">
            <button
              v-for="sample in SAMPLE_TOPICS"
              :key="sample"
              class="sample-chip"
              type="button"
              @click="fillSample(sample)"
            >
              {{ sample }}
            </button>
          </div>

          <div class="action-row">
            <button
              class="primary-button"
              :data-tone="primaryActionTone"
              type="button"
              :disabled="primaryActionDisabled"
              @click="handlePrimaryAction"
            >
              <component :is="primaryActionIcon" :size="16" />
              <span>{{ primaryActionLabel }}</span>
            </button>
          </div>
        </form>

        <div class="recent-panel">
          <div class="subhead">
            <h3>最近任务</h3>
            <span>{{ recentJobCards.length }}</span>
          </div>

          <button
            v-for="job in recentJobCards"
            :key="job.job_id"
            class="recent-item"
            type="button"
            @click="openRecent(job.job_id)"
          >
            <div>
              <p>{{ job.topic }}</p>
              <span>{{ statusLabel(job.status) }} · {{ job.review_score }}/10</span>
            </div>
            <ArrowRight :size="14" />
          </button>
        </div>
      </section>

      <section class="panel flow-panel">
        <div class="panel-head">
          <div>
            <p class="section-kicker">执行流</p>
            <h2>任务进度</h2>
          </div>
          <div class="job-meta">
            <span>{{ currentJob?.status ? statusLabel(currentJob.status) : "空闲" }}</span>
            <span>{{ totalTokens.toLocaleString() }} Token</span>
          </div>
        </div>

        <div v-if="currentJob" class="summary-strip">
          <div>
            <p class="summary-label">主题</p>
            <strong>{{ currentJob.topic }}</strong>
          </div>
          <div>
            <p class="summary-label">当前阶段</p>
            <strong>{{ stepLabel(currentJob.current_step) }}</strong>
          </div>
          <div>
            <p class="summary-label">更新时间</p>
            <strong>{{ formatTime(currentJob.finished_at || currentJob.started_at || currentJob.created_at) }}</strong>
          </div>
        </div>

        <div class="stage-grid">
          <article v-for="stage in stageCards" :key="stage.key" class="stage-card" :data-state="stage.state">
            <div class="stage-head">
              <component :is="stage.icon" :size="18" />
              <span>{{ stage.label }}</span>
              <em>{{ stageLabel(stage.state) }}</em>
            </div>
            <strong class="stage-value">{{ stage.value }}</strong>
            <p class="stage-hint">{{ stage.hint }}</p>
          </article>
        </div>

        <div class="insight-grid" v-if="currentJob">
          <article class="insight-card">
            <div class="subhead">
              <h3>子问题</h3>
              <span>{{ researchPlan.length }}</span>
            </div>
            <div class="plan-list">
              <article v-if="!researchPlan.length" class="empty-block">
                等待规划阶段输出子问题。
              </article>
              <div v-for="(item, index) in researchPlan" :key="`${index}-${item}`" class="plan-item">
                <span class="plan-index">{{ String(index + 1).padStart(2, "0") }}</span>
                <p>{{ item }}</p>
              </div>
            </div>
          </article>

          <article class="insight-card">
            <div class="subhead">
              <h3>审核建议</h3>
              <span>{{ currentJob.review_passed ? "通过" : "待更新" }}</span>
            </div>
            <div class="feedback-box" :class="{ empty: !reviewFeedback }">
              <template v-if="reviewFeedback">
                <p>{{ reviewFeedback }}</p>
              </template>
              <template v-else>
                <p>审核完成后，这里会显示修改建议。</p>
              </template>
            </div>
          </article>
        </div>

        <div class="timeline-block">
          <div class="subhead">
            <h3>事件日志</h3>
            <span>{{ events.length }}</span>
          </div>

          <div class="timeline-list">
            <article v-if="!events.length" class="empty-block">
              等待第一个任务事件。
            </article>
            <article v-for="event in events.slice(-18)" :key="event.seq" class="timeline-item">
              <div class="timeline-top">
                <span class="event-type">{{ formatEventType(event.type) }}</span>
                <span class="event-time">{{ formatTime(event.timestamp) }}</span>
              </div>
              <p>{{ formatEventMessage(event) }}</p>
            </article>
          </div>
        </div>

        <div class="tokens-block" v-if="currentTokenRows.length">
          <div class="subhead">
            <h3>Token 用量</h3>
            <span>{{ totalTokens.toLocaleString() }}</span>
          </div>
          <div class="token-table">
            <div class="token-row token-head">
              <span>节点</span>
              <span>输入</span>
              <span>输出</span>
              <span>合计</span>
            </div>
            <div v-for="row in currentTokenRows" :key="row.node" class="token-row">
              <span>{{ tokenNodeLabel(row.node) }}</span>
              <span>{{ row.prompt }}</span>
              <span>{{ row.completion }}</span>
              <span>{{ row.total }}</span>
            </div>
          </div>
        </div>
      </section>
    </main>

    <section class="panel result-panel">
      <div class="panel-head">
        <div>
          <p class="section-kicker">产出</p>
          <h2>报告与证据</h2>
        </div>
        <div class="tabs">
          <button
            v-for="tab in TABS"
            :key="tab.id"
            class="tab-button"
            :class="{ active: activeTab === tab.id }"
            type="button"
            @click="activeTab = tab.id"
          >
            {{ tab.label }}
          </button>
        </div>
      </div>

      <div class="result-scroll">
        <div class="report-meta" v-if="currentJob">
          <div>
            <p class="summary-label">评分</p>
            <strong>{{ currentJob.review_score }}/10</strong>
          </div>
          <div>
            <p class="summary-label">状态</p>
            <strong>{{ statusLabel(currentJob.status) }}</strong>
          </div>
          <div>
            <p class="summary-label">修订次数</p>
            <strong>{{ currentJob.revision_count }}</strong>
          </div>
        </div>

        <div v-if="activeTab === 'report'" class="report-body">
          <div v-if="reportHtml" class="markdown" v-html="reportHtml" />
          <div v-else class="empty-block">
            暂无报告。发起研究后，报告会显示在这里。
          </div>
        </div>

        <div v-else-if="activeTab === 'evidence'" class="evidence-list">
          <article v-if="!currentJob?.evidence_pack?.length" class="empty-block">
            研究节点完成后，证据会显示在这里。
          </article>
          <article v-for="item in currentJob?.evidence_pack || []" :key="item.evidence_id" class="evidence-item">
            <div class="evidence-head">
              <span class="evidence-id">{{ item.evidence_id }}</span>
              <span class="evidence-kind">{{ item.kind }}</span>
            </div>
            <strong>{{ item.title || item.source || "未命名来源" }}</strong>
            <p>{{ item.text?.slice(0, 260) }}</p>
            <a v-if="item.url" :href="item.url" target="_blank" rel="noreferrer">{{ item.url }}</a>
          </article>

          <div v-if="currentJob?.citations?.length" class="citation-block">
            <div class="subhead">
              <h3>引用来源</h3>
              <span>{{ currentJob.citations.length }}</span>
            </div>
            <article v-for="cite in currentJob.citations" :key="cite.index" class="citation-item">
              <span>[{{ cite.index }}]</span>
              <div>
                <strong>{{ cite.title }}</strong>
                <p>{{ cite.url }}</p>
              </div>
            </article>
          </div>
        </div>

        <div v-else class="raw-block">
          <pre>{{ JSON.stringify({ job: currentJob, report: currentReport }, null, 2) }}</pre>
        </div>
      </div>
    </section>
  </div>
</template>
