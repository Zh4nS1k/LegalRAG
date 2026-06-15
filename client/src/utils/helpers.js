// helpers.js
export function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function splitAnalysisIntoSections(htmlContent) {
  const sections = {
    risks: '',
    recommendations: '',
    summary: '',
  };

  // Create temporary DOM element to parse HTML
  const parser = new DOMParser();
  const doc = parser.parseFromString(htmlContent, 'text/html');

  // Get all heading elements
  const headings = doc.querySelectorAll('h1, h2, h3, h4, h5, h6');
  let currentSection = null;

  headings.forEach((heading) => {
    const text = heading.textContent.toLowerCase();

    if (
      text.includes('правовые риски') ||
      text.includes('неясные формулировки') ||
      text.includes('возможные нарушения') ||
      text.includes('риски') ||
      text.includes('проблемные места') ||
      text.includes('несоответствия')
    ) {
      currentSection = 'risks';
    } else if (
      text.includes('рекомендации') ||
      text.includes('улучшения') ||
      text.includes('исправления')
    ) {
      currentSection = 'recommendations';
    } else if (
      text.includes('заключение') ||
      text.includes('сводка') ||
      text.includes('итог') ||
      text.includes('вывод') ||
      text.includes('резюме')
    ) {
      currentSection = 'summary';
    }

    if (currentSection) {
      // Get content until next heading of same or higher level
      let content = '';
      let nextElement = heading.nextElementSibling;
      const headingLevel = parseInt(heading.tagName.substring(1));

      while (nextElement) {
        const nextHeadingLevel = nextElement.tagName.match(/^H(\d)$/);
        if (nextHeadingLevel && parseInt(nextHeadingLevel[1]) <= headingLevel) {
          break;
        }
        content += nextElement.outerHTML;
        nextElement = nextElement.nextElementSibling;
      }

      if (sections[currentSection]) {
        sections[currentSection] +=
          '<hr style="margin: 20px 0; border-color: #eee;"/>';
      }
      sections[currentSection] += heading.outerHTML + content;
    }
  });

  // If no sections found, use the whole content for full analysis
  if (!sections.risks && !sections.recommendations && !sections.summary) {
    sections.risks = htmlContent;
  }

  return sections;
}
