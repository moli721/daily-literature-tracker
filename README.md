<p align="center">
  <img src="./assets/LOGO.png" alt="EvoMaster Logo" width="" />
</p>

<p align="center">
  <strong>【<a href="./README.md">English</a> | <a href="./README-zh.md">简体中文</a>】</strong>
</p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick%20Start-%7E3%20min-0ea5e9?style=for-the-badge" alt="Quick Start"></a>
  <a href="#scimaster-ecosystem"><img src="https://img.shields.io/badge/SciMaster-6%2B%20Agents-059669?style=for-the-badge" alt="SciMaster"></a>
  <a href="#key-features"><img src="https://img.shields.io/badge/Key%20Features-4%20Highlights-7c3aed?style=for-the-badge" alt="Key Features"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-ea580c?style=for-the-badge" alt="License"></a>
</p>

<div align="center">

**A general-purpose agent foundation for Autonomous Scientific Research**

*Making scientific agent development simpler, modular, and scalable—accelerating AI for Science adoption.*

<table align="center" width="100%">
<tr>
<td width="33%" align="center" valign="top">

**LLM training**

https://github.com/user-attachments/assets/62c132c1-6fe8-4c18-89c6-be330fab2c6f

</td>
<td width="33%" align="center" valign="top">

**Material science**

https://github.com/user-attachments/assets/590365c0-95a6-467e-a22b-3c373fb2bb8a

</td>
<td width="33%" align="center" valign="top">

**Create an ML agent**

https://github.com/user-attachments/assets/d5e2500b-f589-4676-b6cb-dce8ae000f2c

</td>
</tr>
</table>

</div>

---

## Contents

- [Introduction](#introduction)
- [Key features](#key-features)
- [SciMaster ecosystem](#scimaster-ecosystem)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [Contributing](#contributing)

---

## <span id="introduction">📖 Introduction</span>

**MagiClaw** is a Feishu (Lark) intelligent assistant: describe what you need in chat, and it orchestrates specialized agents for complex tasks; when needed, it can also help you design and generate new agents on the [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) framework.

**EvoMaster** is the lightweight agent framework underneath: it handles tool calling, skills, memory, sessions, and playground wiring so you can focus on behavior design and prompts instead of rebuilding engineering plumbing.

---

## <span id="key-features">✨ Key features</span>

### 1. 💬 Native Feishu / Lark experience

Talk to the bot like a teammate: multi-turn context, card interactions, and optional Feishu document-related capabilities that match how teams already collaborate.

### 2. 🔀 Orchestration & delegation

The default **magiclaw** can delegate work to other registered playgrounds (e.g. **agent_builder**) via tools, so one conversation can drive multi-step or meta-level (agent-building) flows.

### 3. 🧰 Rich tool surface

Configure **MCP**, web search and fetch, optional Feishu read / file send, **Skills**, and persistent **memory** in `configs/magiclaw/config.yaml`.

### 4. 🏗️ Built on EvoMaster

Aligned with the EvoMaster ecosystem: pluggable LLM backends, local or container sessions, and a small codebase that is easy to extend.

### <span id="scimaster-ecosystem">SciMaster ecosystem</span>

The full **SciMaster** line (ML-Master, X-Master, Browse-Master, etc.) lives in the upstream [EvoMaster](https://github.com/sjtu-sai-agents/EvoMaster) repository. This repo focuses on the Feishu assistant and the `agent_builder` path; sync or wire other playgrounds from EvoMaster as needed.

---

## <span id="roadmap">🗺️ Roadmap (high level)</span>

| Phase | Focus |
|-------|--------|
| **Now** | MagiClaw Feishu bot, `magiclaw` + `agent_builder`, core `evomaster` library |
| **Next** | Better support for Feishu CLI and related tooling, broader Feishu capabilities, and a smoother deployment experience |

---

## <span id="repository-layout">🏗️ Repository layout</span>

```
MagiClaw/
├── evomaster/              # Core library (agent, core, interface/feishu, memory, skills, …)
├── playground/
│   ├── magiclaw/         # Default Feishu conversational agent
│   └── agent_builder/      # Meta-agent: design / generate agents
├── configs/
│   ├── feishu/             # Bot connection & credentials
│   ├── magiclaw/         # LLM, tools, memory, MCP
│   └── agent_builder/      # Planner + builder agents
├── run.py                  # Optional: run a playground from the CLI
├── requirements.txt
└── pyproject.toml
```

---

## <span id="quick-start">🚀 Quick start</span>

### Requirements

- **Python** ≥ 3.12

### Install

```bash
git clone https://github.com/sjtu-sai-agents/MagiClaw.git
cd MagiClaw
pip install -r requirements.txt
```

With [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -r requirements.txt
# Or, if you use project metadata: uv sync
```

### 1. Create a Feishu app and bot

1. Open the [Feishu Open Platform](https://open.feishu.cn/app) and sign in.  
2. Create an application and enable the **bot** capability as guided.

### 2. App credentials (`.env`)

Copy `.env.template` in the project root to `.env`. From the Feishu developer console, copy the bot **App ID** and **App Secret** into:

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`

### 3. Import scopes

In the console, go to **Permissions & Scopes** → **Batch import/export scopes**, and import the following JSON (trim scopes to what you actually need):

```json
{
  "scopes": {
    "tenant": [
      "im:resource",
      "docx:document",
      "docx:document:readonly",
      "drive:drive",
      "im:chat:readonly",
      "im:message",
      "im:message.group_at_msg:readonly",
      "im:message.group_msg",
      "im:message.p2p_msg:readonly",
      "im:message:readonly",
      "im:message:recall",
      "im:message:send_as_bot",
      "wiki:wiki:readonly"
    ],
    "user": [
      "drive:drive",
      "drive:drive.metadata:readonly",
      "drive:drive.search:readonly",
      "drive:drive:readonly",
      "drive:drive:version",
      "drive:drive:version:readonly"
    ]
  }
}
```

### 4. Event subscription (long connection)

**Events & callbacks** → **Event configuration** → choose **Receive events through persistent connection**, and add:

| Description | Event name |
|-------------|------------|
| Bot added to group v2.0 | `im.chat.member.bot.added_v1` |
| Bot removed from group v2.0 | `im.chat.member.bot.deleted_v1` |
| Message read v2.0 | `im.message.message_read_v1` |
| Message.recalled v2.0 | `im.message.recalled_v1` |
| Message received v2.0 | `im.message.receive_v1` |

### 5. Callback subscription (long connection)

**Events & callbacks** → **Callback configuration** → **Receive callbacks through persistent connection**, subscribe to:

| Description | Callback |
|-------------|----------|
| Card callback communication | `card.action.trigger` |

### 6. Publish a version

Under **Version management & release**, create a version, fill in the details, and **publish** so the bot is available in production.

### 7. LLM and other APIs

Edit `.env` in the project root for LLM, search, and other API keys and endpoints. Default models and agent behavior are in `configs/magiclaw/config.yaml`; Feishu connection settings are in `configs/feishu/config.yaml`.

### 8. Start the Feishu bot

```bash
python -m evomaster.interface.feishu --config configs/feishu/config.yaml
```

When it is running, you can chat with MagiClaw in Feishu.

---

## <span id="contributing">🤝 Contributing</span>

Feedback via [Issues](https://github.com/sjtu-sai-agents/MagiClaw/issues) and improvements via Pull Requests are welcome. For larger changes, please open an issue first to align on scope and design.

This project is licensed under [Apache 2.0](./LICENSE).
