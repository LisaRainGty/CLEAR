# 论文草稿读取说明

- `实验结果与分析.docx`：用户提供的原始完整草稿，未修改。
- `实验结果与分析_完整提取.md`：通过 Pandoc 转换的完整可搜索文本；表格、公式标记和图片链接均保留。
- `media/`：从 DOCX 提取的七张原图。
- `preview/.../Preview.html`：macOS Quick Look 生成的结构预览与附件。

标准 DOCX 页面渲染器在本机因 LibreOffice 动态库 `liblcms2.2.dylib` 缺失而无法启动，因此没有把 Quick Look 预览冒充为严格的 Word 分页渲染。正文与表格审计以 Markdown、DOCX XML 和 Quick Look 结构预览交叉核对；DOCX 内未发现 tracked changes、comments、`w:ins` 或 `w:del`。
