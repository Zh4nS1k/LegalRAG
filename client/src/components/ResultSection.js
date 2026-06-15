// ResultSection.js
import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Button,
  Tabs,
  Tab,
  Box,
  Fade,
} from '@mui/material';
import hljs from 'highlight.js';
import MarkdownIt from 'markdown-it';
import { splitAnalysisIntoSections } from '../utils/helpers';

import 'highlight.js/styles/github.css';

const md = new MarkdownIt({
  html: true,
  linkify: true,
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return `<pre class="hljs"><code>${hljs.highlight(str, { language: lang }).value
          }</code></pre>`;
      } catch (__) { }
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`;
  },
});

function ResultSection({ data, onBackClick }) {
  const [tab, setTab] = useState(0);

  useEffect(() => {
    document.querySelectorAll('pre code').forEach((block) => {
      hljs.highlightElement(block);
    });
  }, [tab]);

  const rawContent = data.analysis || '';
  const htmlContent = md.render(rawContent);
  const sections = splitAnalysisIntoSections(htmlContent);

  const enhanceContentWithStyles = (html) => {
    const container = document.createElement('div');
    container.innerHTML = html;

    // Style risk levels
    container.querySelectorAll('li, p').forEach((el) => {
      const text = el.textContent.toLowerCase();
      if (text.includes('высок') || text.includes('high')) {
        el.classList.add('risk-high');
      } else if (text.includes('средн') || text.includes('medium')) {
        el.classList.add('risk-medium');
      } else if (text.includes('низк') || text.includes('low')) {
        el.classList.add('risk-low');
      }
    });

    // Add icons or badges for legal references
    container.querySelectorAll('li').forEach((el) => {
      if (
        el.textContent.includes('Статья') ||
        el.textContent.includes('Закон')
      ) {
        el.classList.add('legal-reference');
      }
    });

    return container.innerHTML;
  };

  const tabs = [
    {
      label: 'Полный анализ',
      content: enhanceContentWithStyles(htmlContent),
      description: 'Полный анализ',
    },
    {
      label: 'Риски',
      content: sections.risks
        ? enhanceContentWithStyles(sections.risks)
        : '<p class="no-risks">Правовые риски не выявлены.</p>',
      description: 'Выявленные правовые риски и их уровень опасности',
    },
    {
      label: 'Рекомендации',
      content: sections.recommendations
        ? enhanceContentWithStyles(sections.recommendations)
        : '<p class="no-recommendations">Рекомендации отсутствуют.</p>',
      description: 'Предложения по устранению выявленных проблем',
    },
    {
      label: 'Сводка',
      content: sections.summary
        ? enhanceContentWithStyles(sections.summary)
        : '<p class="no-summary">Нет данных для сводки.</p>',
      description: 'Краткое резюме основных проблем и рекомендаций',
    },
  ];

  return (
    <Container sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Результаты юридического анализа
      </Typography>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Box>
          <Typography variant="subtitle1" gutterBottom>
            <strong>Тип документа:</strong> {data.document_type || 'Неизвестно'}
          </Typography>
          <Typography variant="subtitle2" color="text.secondary">
            {new Date().toLocaleDateString()}
          </Typography>
        </Box>
        <Button
          variant="outlined"
          onClick={onBackClick}
          sx={{ height: 'fit-content' }}
        >
          ← Назад к загрузке
        </Button>
      </Box>

      <Tabs
        value={tab}
        onChange={(e, newTab) => setTab(newTab)}
        variant="fullWidth"
        sx={{ mb: 2 }}
      >
        {tabs.map((t, i) => (
          <Tab
            key={i}
            label={
              <Box sx={{ display: 'flex', flexDirection: 'column' }}>
                <span>{t.label}</span>
                <Typography variant="caption" color="text.secondary">
                  {t.description}
                </Typography>
              </Box>
            }
          />
        ))}
      </Tabs>

      <Fade in key={tab}>
        <Box
          sx={{
            mt: 1,
            p: 3,
            backgroundColor: '#fff',
            borderRadius: 2,
            boxShadow: 1,
            minHeight: '400px',
            '& .risk-high': {
              borderLeft: '4px solid #E60000',
              paddingLeft: '12px',
              backgroundColor: '#fff5f5',
            },
            '& .risk-medium': {
              borderLeft: '4px solid #666666',
              paddingLeft: '12px',
              backgroundColor: '#F3F4F6',
            },
            '& .risk-low': {
              borderLeft: '4px solid #000000',
              paddingLeft: '12px',
              backgroundColor: '#F9FAFB',
            },
            '& .legal-reference': {
              fontWeight: '600',
              color: '#CC0000',
            },
            '& .no-risks, & .no-recommendations, & .no-summary': {
              fontStyle: 'italic',
              color: '#666666',
            },
          }}
          className="analysis-container"
          dangerouslySetInnerHTML={{ __html: tabs[tab].content }}
        />
      </Fade>
    </Container>
  );
}

export default ResultSection;
