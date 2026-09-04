import { marked } from "marked";
import DOMPurify from "dompurify";

marked.setOptions({
  breaks: true,
  gfm: true
});

export function renderMarkdown(source = "") {
  if (!source) return "";
  return DOMPurify.sanitize(marked.parse(source));
}
