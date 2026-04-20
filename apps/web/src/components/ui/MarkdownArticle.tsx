import { Fragment, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type RenderCitation = (index: number) => ReactNode;

type MarkdownArticleProps = {
  content: string;
  renderCitation?: RenderCitation;
  renderCodeActions?: (code: string) => ReactNode;
};

function replaceCitationsInChildren(children: ReactNode, renderCitation: RenderCitation): ReactNode {
  const input = Array.isArray(children) ? children : [children];
  const output: ReactNode[] = [];
  const pattern = /\[(\d+)\]/g;

  input.forEach((child, childIndex) => {
    if (typeof child !== "string") {
      output.push(child);
      return;
    }

    let cursor = 0;
    let match: RegExpExecArray | null = null;
    let localIndex = 0;
    pattern.lastIndex = 0;

    while ((match = pattern.exec(child)) !== null) {
      if (match.index > cursor) {
        output.push(child.slice(cursor, match.index));
      }
      const parsed = Number.parseInt(match[1] ?? "", 10);
      if (Number.isFinite(parsed) && parsed > 0) {
        output.push(
          <Fragment key={`citation-${childIndex}-${localIndex}`}>{renderCitation(parsed - 1)}</Fragment>,
        );
      } else {
        output.push(match[0]);
      }
      localIndex += 1;
      cursor = match.index + match[0].length;
    }

    if (cursor < child.length) {
      output.push(child.slice(cursor));
    }
  });

  return output;
}

function extractText(node: ReactNode): string {
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(extractText).join("");
  }
  if (node && typeof node === "object" && "props" in node) {
    return extractText((node as { props?: { children?: ReactNode } }).props?.children ?? "");
  }
  return "";
}

export function MarkdownArticle({ content, renderCitation, renderCodeActions }: MarkdownArticleProps) {
  const applyCitations = (children: ReactNode): ReactNode =>
    renderCitation ? replaceCitationsInChildren(children, renderCitation) : children;

  return (
    <article className="markdown-article">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{applyCitations(children)}</p>,
          li: ({ children }) => <li>{applyCitations(children)}</li>,
          h1: ({ children }) => <h1>{applyCitations(children)}</h1>,
          h2: ({ children }) => <h2>{applyCitations(children)}</h2>,
          h3: ({ children }) => <h3>{applyCitations(children)}</h3>,
          h4: ({ children }) => <h4>{applyCitations(children)}</h4>,
          h5: ({ children }) => <h5>{applyCitations(children)}</h5>,
          h6: ({ children }) => <h6>{applyCitations(children)}</h6>,
          td: ({ children }) => <td>{applyCitations(children)}</td>,
          th: ({ children }) => <th>{applyCitations(children)}</th>,
          blockquote: ({ children }) => <blockquote>{applyCitations(children)}</blockquote>,
          table: ({ children }) => (
            <div className="markdown-article__table-wrap">
              <table className="markdown-article__table">{children}</table>
            </div>
          ),
          pre: ({ children }) => {
            const codeText = extractText(children).trim();
            return (
              <div className="markdown-article__code-shell">
                {renderCodeActions && codeText ? (
                  <div className="markdown-article__code-actions">{renderCodeActions(codeText)}</div>
                ) : null}
                <pre className="markdown-article__code">{children}</pre>
              </div>
            );
          },
          a: ({ href, children, ...rest }) => (
            <a href={href} target="_blank" rel="noreferrer noopener" {...rest}>
              {children}
            </a>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </article>
  );
}


