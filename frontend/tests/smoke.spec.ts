import { test, expect } from "@playwright/test"

test("网页壳可以加载并显示对话空状态", async ({ page }) => {
  await page.route("**/api/history", async (route) => {
    await route.fulfill({ json: { messages: [] } })
  })
  await page.route("**/api/sessions", async (route) => {
    await route.fulfill({
      json: {
        currentSessionId: "current",
        sessions: [{
          id: "current",
          createdAt: "2026-01-01T00:00:00",
          updatedAt: "2026-01-01T00:00:00",
          messageCount: 0,
          title: "空会话",
          active: true,
        }],
      },
    })
  })
  await page.route("**/api/reset", async (route) => {
    await route.fulfill({ json: { messages: [] } })
  })
  await page.goto("/")
  await expect(page.getByText("输入一个问题，开始学习。"))
    .toBeVisible()
  await expect(page.locator("textarea")).toBeVisible()
})

test("桌面端滚动研究正文时会话栏固定在视窗内", async ({ page }) => {
  const longAnswer = Array.from({ length: 80 }, (_, index) => `第 ${index + 1} 行测试内容`).join("\n")
  await page.route("**/api/history", async (route) => {
    await route.fulfill({
      json: { messages: [{ id: "a-long", role: "assistant", content: longAnswer }] },
    })
  })
  await page.route("**/api/sessions", async (route) => {
    await route.fulfill({
      json: {
        currentSessionId: "current",
        sessions: [{
          id: "current",
          createdAt: "2026-01-01T00:00:00",
          updatedAt: "2026-01-01T00:00:00",
          messageCount: 1,
          title: "长对话",
          active: true,
        }],
      },
    })
  })

  await page.goto("/")
  const sessionPanel = page.getByLabel("会话管理")
  const initialBox = await sessionPanel.boundingBox()
  expect(initialBox).not.toBeNull()

  const conversationBody = page.locator(".conversation-body")
  await conversationBody.evaluate((element) => element.scrollTo(0, element.scrollHeight))
  await expect.poll(() => conversationBody.evaluate((element) => element.scrollTop)).toBeGreaterThan(0)

  const scrolledBox = await sessionPanel.boundingBox()
  expect(scrolledBox).not.toBeNull()
  expect(Math.abs(scrolledBox!.y - initialBox!.y)).toBeLessThanOrEqual(1)
  expect(scrolledBox!.y).toBeGreaterThan(0)
})

test("网页壳可以通过 SSE 发送并显示一轮消息", async ({ page }) => {
  await page.route("**/api/history", async (route) => {
    await route.fulfill({ json: { messages: [] } })
  })
  await page.route("**/api/sessions", async (route) => {
    await route.fulfill({
      json: {
        currentSessionId: "current",
        sessions: [{
          id: "current",
          createdAt: "2026-01-01T00:00:00",
          updatedAt: "2026-01-01T00:00:00",
          messageCount: 0,
          title: "空会话",
          active: true,
        }],
      },
    })
  })
  await page.route("**/api/chat", async (route) => {
    await route.fulfill({
      headers: { "content-type": "text/event-stream; charset=utf-8" },
      body: [
        'event: message\ndata: {"message":{"id":"u1","role":"user","content":"测试问题"}}\n\n',
        'event: message\ndata: {"message":{"id":"a1","role":"assistant","content":"测试回答"}}\n\n',
        'event: done\ndata: {"messages":[{"id":"u1","role":"user","content":"测试问题"},{"id":"a1","role":"assistant","content":"测试回答"}]}\n\n',
      ].join(""),
    })
  })

  await page.goto("/")
  const input = page.locator("textarea")
  await input.fill("测试问题")
  await input.press("Enter")

  await expect(page.getByText("测试回答")).toBeVisible()
  await expect(input).toHaveValue("")
})

test("会话工作台可以搜索历史并查看统计", async ({ page }) => {
  await page.route("**/api/history", async (route) => {
    await route.fulfill({ json: { messages: [] } })
  })
  await page.route("**/api/sessions", async (route) => {
    await route.fulfill({
      json: {
        currentSessionId: "current",
        sessions: [
          {
            id: "current",
            createdAt: "2026-08-25T00:00:00",
            updatedAt: "2026-08-25T00:00:00",
            messageCount: 0,
            title: "空会话",
            active: true,
          },
          {
            id: "archive-1",
            createdAt: "2026-08-24T00:00:00",
            updatedAt: "2026-08-24T00:00:00",
            messageCount: 4,
            title: "世界杯研究",
            active: false,
          },
        ],
      },
    })
  })
  await page.route("**/api/sessions/search?*", async (route) => {
    await route.fulfill({ json: { hits: [{ sessionId: "archive-1", index: 2, content: "世界杯比赛资料" }] } })
  })
  await page.route("**/api/sessions/stats", async (route) => {
    await route.fulfill({
      json: {
        stats: [
          { sessionId: "current", messageCount: 0, inputTokens: 0, outputTokens: 0, cacheReadTokens: 0, estimatedTokens: 0 },
          { sessionId: "archive-1", messageCount: 4, inputTokens: 180, outputTokens: 20, cacheReadTokens: 0, estimatedTokens: 200 },
        ],
      },
    })
  })

  await page.goto("/")
  await page.getByLabel("搜索会话").fill("世界杯")
  await expect(page.getByText(/#2 世界杯比赛资料/)).toBeVisible()
  await page.getByRole("button", { name: "统计" }).click()
  await expect(page.getByText("估算 200 Token")).toBeVisible()
})

test("工作流命令和进度按 SSE 到达顺序插入对话时间线", async ({ page }) => {
  await page.route("**/api/history", async (route) => {
    await route.fulfill({ json: { messages: [] } })
  })
  await page.route("**/api/sessions", async (route) => {
    await route.fulfill({
      json: {
        currentSessionId: "current",
        sessions: [{
          id: "current",
          createdAt: "2026-01-01T00:00:00",
          updatedAt: "2026-01-01T00:00:00",
          messageCount: 0,
          title: "空会话",
          active: true,
        }],
      },
    })
  })
  let commandText = ""
  await page.route("**/api/command", async (route) => {
    commandText = (route.request().postDataJSON() as { text: string }).text
    await route.fulfill({
      headers: { "content-type": "text/event-stream; charset=utf-8" },
      body: [
        'event: command\ndata: {"command":"workflow","detail":"有界协作工作流已启动","messages":[] }\n\n',
        'event: message\ndata: {"message":{"id":"u1","role":"user","content":"研究项目"}}\n\n',
        'event: workflow\ndata: {"workflow":{"phase":"reviewing","status":"running","detail":"审核中","role":"reviewer","usage":{"roleInvocations":2,"modelRequests":2,"toolCalls":0,"inputTokens":0,"outputTokens":0,"cacheReadTokens":0,"delegationCalls":0,"elapsedSeconds":0}}}\n\n',
        'event: message\ndata: {"message":{"id":"a1","role":"assistant","content":"最终报告"}}\n\n',
        'event: done\ndata: {"messages":[{"id":"u1","role":"user","content":"研究项目"},{"id":"a1","role":"assistant","content":"最终报告"}],"workflow":{"phase":"completed","status":"passed","detail":"通过"}}\n\n',
      ].join(""),
    })
  })

  await page.goto("/")
  const input = page.locator("textarea")
  await page.getByRole("button", { name: "多 Agent 工作流" }).click()
  await input.fill("研究项目")
  await input.press("Enter")

  const command = page.locator('[data-command="workflow"]')
  const user = page.getByText("研究项目")
  const workflow = page.getByLabel("协作工作流")
  const assistant = page.getByText("最终报告")
  await expect(assistant).toBeVisible()
  const order = await page.evaluate(() => {
    const nodes = [
      document.querySelector('[data-command="workflow"]'),
      document.querySelector('[data-message-id="u1"]'),
      document.querySelector('[aria-label="协作工作流"]'),
      document.querySelector('[data-message-id="a1"]'),
    ]
    return nodes.every((node, index) => node && (index === 0 || Boolean(nodes[index - 1]!.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING)))
  })
  expect(order).toBe(true)
  await expect(command).toBeVisible()
  await expect(user).toBeVisible()
  await expect(workflow).toBeVisible()
  expect(commandText).toBe("/workflow 研究项目")
})
