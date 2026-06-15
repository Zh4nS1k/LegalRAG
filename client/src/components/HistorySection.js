import React, { useEffect, useState } from 'react';
import {
  Container,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  CircularProgress,
  Alert,
  Box,
  Collapse,
  IconButton,
  Tabs,
  Tab,
  Fade,
  Chip,
  useTheme,
  alpha,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DescriptionIcon from '@mui/icons-material/Description';
import EventIcon from '@mui/icons-material/Event';
import CategoryIcon from '@mui/icons-material/Category';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import RiskHighIcon from '@mui/icons-material/Warning';
import SummaryIcon from '@mui/icons-material/Summarize';
import RecommendationIcon from '@mui/icons-material/TipsAndUpdates';
import hljs from 'highlight.js';
import MarkdownIt from 'markdown-it';
import { splitAnalysisIntoSections } from '../utils/helpers';
import { motion } from 'framer-motion';

import 'highlight.js/styles/github.css';

const md = new MarkdownIt({
  html: true,
  linkify: true,
  highlight: function (str, lang) {
    if (lang && hljs.getLanguage(lang)) {
      try {
        return `<pre class="hljs"><code>${
          hljs.highlight(str, { language: lang }).value
        }</code></pre>`;
      } catch (__) {}
    }
    return `<pre class="hljs"><code>${md.utils.escapeHtml(str)}</code></pre>`;
  },
});

const tabIcons = {
  0: <DescriptionIcon fontSize="small" />,
  1: <RiskHighIcon fontSize="small" />,
  2: <RecommendationIcon fontSize="small" />,
  3: <SummaryIcon fontSize="small" />,
};

const HistoryItem = ({ item }) => {
  const theme = useTheme();
  const [expanded, setExpanded] = useState(false);
  const [currentTab, setCurrentTab] = useState(0);

  const toggleExpand = () => {
    setExpanded(!expanded);
  };

  const enhanceContentWithStyles = (html) => {
    const container = document.createElement('div');
    container.innerHTML = html;

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

  const processAnalysisContent = (analysis) => {
    if (!analysis)
      return {
        tabs: [
          {
            label: 'Анализ',
            content: '<p class="no-content">Нет данных анализа</p>',
            description: 'Полный текст',
          },
        ],
        sections: {},
      };

    const htmlContent = md.render(analysis);
    const sections = splitAnalysisIntoSections(htmlContent);

    return {
      tabs: [
        {
          label: 'Анализ',
          content: enhanceContentWithStyles(htmlContent),
          description: 'Полный текст',
          icon: tabIcons[0],
        },
        {
          label: 'Риски',
          content: sections.risks
            ? enhanceContentWithStyles(sections.risks)
            : '<p class="no-risks">Риски не выявлены</p>',
          description: 'Уровни опасности',
          icon: tabIcons[1],
        },
        {
          label: 'Рекомендации',
          content: sections.recommendations
            ? enhanceContentWithStyles(sections.recommendations)
            : '<p class="no-recommendations">Нет рекомендаций</p>',
          description: 'Советы',
          icon: tabIcons[2],
        },
        {
          label: 'Сводка',
          content: sections.summary
            ? enhanceContentWithStyles(sections.summary)
            : '<p class="no-summary">Нет сводки</p>',
          description: 'Основное',
          icon: tabIcons[3],
        },
      ],
      sections,
    };
  };

  const { tabs } = processAnalysisContent(item.analysis);

  return (
    <Grid item xs={12}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <Card
          elevation={0}
          sx={{
            mb: 2,
            borderRadius: 3,
            backgroundColor: expanded
              ? '#fff'
              : alpha(theme.palette.background.paper, 0.8),
            border: `1px solid ${
              expanded ? theme.palette.divider : 'transparent'
            }`,
            overflow: 'hidden',
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
            backdropFilter: 'blur(10px)',
            '&:hover': {
              backgroundColor: '#fff',
              borderColor: theme.palette.divider,
            },
          }}
        >
          <CardContent sx={{ p: 0 }}>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                p: 2,
                cursor: 'pointer',
                transition: 'background-color 0.3s',
              }}
              onClick={toggleExpand}
              component={motion.div}
              whileTap={{ scale: 0.98 }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: 2,
                    backgroundColor: alpha(theme.palette.primary.main, 0.1),
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <DescriptionIcon
                    sx={{
                      fontSize: 20,
                      color: theme.palette.primary.main,
                    }}
                  />
                </Box>
                <Box>
                  <Typography
                    variant="subtitle1"
                    sx={{
                      fontWeight: 500,
                    }}
                  >
                    {item.filename}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                    <Chip
                      icon={<CategoryIcon sx={{ fontSize: 14 }} />}
                      label={item.type || 'Неизвестно'}
                      size="small"
                      variant="outlined"
                      sx={{
                        borderRadius: 1.5,
                        fontSize: 12,
                        height: 24,
                        backgroundColor: alpha(
                          theme.palette.background.default,
                          0.5
                        ),
                      }}
                    />
                    <Chip
                      icon={<EventIcon sx={{ fontSize: 14 }} />}
                      label={new Date(item.created_at).toLocaleDateString(
                        'ru-RU',
                        {
                          day: 'numeric',
                          month: 'short',
                        }
                      )}
                      size="small"
                      variant="outlined"
                      sx={{
                        borderRadius: 1.5,
                        fontSize: 12,
                        height: 24,
                        backgroundColor: alpha(
                          theme.palette.background.default,
                          0.5
                        ),
                      }}
                    />
                  </Box>
                </Box>
              </Box>
              <motion.div
                animate={{ rotate: expanded ? 180 : 0 }}
                transition={{ duration: 0.3 }}
              >
                <IconButton
                  onClick={(e) => {
                    e.stopPropagation();
                    toggleExpand();
                  }}
                  size="small"
                  sx={{
                    color: theme.palette.text.secondary,
                    backgroundColor: alpha(theme.palette.action.active, 0.05),
                    '&:hover': {
                      backgroundColor: alpha(theme.palette.action.active, 0.1),
                    },
                  }}
                >
                  <ExpandMoreIcon fontSize="small" />
                </IconButton>
              </motion.div>
            </Box>

            <Collapse in={expanded}>
              <Box sx={{ px: 2, pb: 2 }}>
                <Tabs
                  value={currentTab}
                  onChange={(e, newTab) => setCurrentTab(newTab)}
                  variant="scrollable"
                  scrollButtons={false}
                  sx={{
                    mb: 2,
                    '& .MuiTabs-indicator': {
                      backgroundColor: theme.palette.primary.main,
                      height: 2,
                      borderRadius: 1,
                    },
                  }}
                >
                  {tabs.map((tab, index) => (
                    <Tab
                      key={index}
                      icon={tabIcons[index]}
                      iconPosition="start"
                      label={
                        <Box
                          sx={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'flex-start',
                            ml: 0.5,
                          }}
                        >
                          <Typography
                            variant="caption"
                            sx={{ fontWeight: 500 }}
                          >
                            {tab.label}
                          </Typography>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{ fontSize: 11 }}
                          >
                            {tab.description}
                          </Typography>
                        </Box>
                      }
                      sx={{
                        textTransform: 'none',
                        minHeight: 48,
                        minWidth: 'unset',
                        px: 1.5,
                        py: 0.5,
                        fontSize: 12,
                        '&.Mui-selected': {
                          color: theme.palette.primary.main,
                        },
                      }}
                      disableRipple
                    />
                  ))}
                </Tabs>

                <Fade in timeout={300}>
                  <Box
                    sx={{
                      p: 2,
                      backgroundColor: theme.palette.background.default,
                      borderRadius: 2,
                      '& .risk-high': {
                        borderLeft: `3px solid ${theme.palette.error.main}`,
                        paddingLeft: 2,
                        backgroundColor: alpha(theme.palette.error.main, 0.05),
                      },
                      '& .risk-medium': {
                        borderLeft: `3px solid ${theme.palette.warning.main}`,
                        paddingLeft: 2,
                        backgroundColor: alpha(
                          theme.palette.warning.main,
                          0.05
                        ),
                      },
                      '& .risk-low': {
                        borderLeft: `3px solid ${theme.palette.success.main}`,
                        paddingLeft: 2,
                        backgroundColor: alpha(
                          theme.palette.success.main,
                          0.05
                        ),
                      },
                      '& .legal-reference': {
                        fontWeight: 500,
                        color: theme.palette.info.main,
                      },
                      '& .no-risks, & .no-recommendations, & .no-summary, & .no-content':
                        {
                          fontStyle: 'italic',
                          color: theme.palette.text.secondary,
                          fontSize: 14,
                        },
                      '& pre': {
                        backgroundColor: alpha(
                          theme.palette.primary.main,
                          0.03
                        ),
                        borderRadius: 1,
                        p: 2,
                        overflowX: 'auto',
                        border: `1px solid ${alpha(
                          theme.palette.divider,
                          0.5
                        )}`,
                        fontSize: 13,
                      },
                      '& code': {
                        fontFamily:
                          'Menlo, Monaco, Consolas, "Courier New", monospace',
                        fontSize: '0.85rem',
                      },
                      '& h1, & h2, & h3': {
                        color: theme.palette.text.primary,
                        marginTop: '0.8em',
                        marginBottom: '0.4em',
                        fontWeight: 600,
                      },
                      '& ul, & ol': {
                        paddingLeft: 2,
                      },
                      '& li': {
                        marginBottom: '0.4em',
                        fontSize: 14,
                        lineHeight: 1.5,
                      },
                      '& p': {
                        fontSize: 14,
                        lineHeight: 1.6,
                        marginBottom: '0.8em',
                      },
                    }}
                    dangerouslySetInnerHTML={{
                      __html: tabs[currentTab].content,
                    }}
                  />
                </Fade>
              </Box>
            </Collapse>
          </CardContent>
        </Card>
      </motion.div>
    </Grid>
  );
};

function HistorySection({ onBackClick }) {
  const theme = useTheme();
  const [historyItems, setHistoryItems] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const token = localStorage.getItem('token');
        const response = await fetch('http://localhost:8080/api/history', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });

        if (!response.ok) {
          throw new Error('Ошибка при загрузке истории');
        }

        const data = await response.json();
        setHistoryItems(data || []);
      } catch (err) {
        setError(err.message);
        setHistoryItems([]);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: '60vh',
        }}
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        >
          <CircularProgress
            size={40}
            thickness={4}
            sx={{ color: theme.palette.primary.main }}
          />
        </motion.div>
      </Box>
    );
  }

  if (error) {
    return (
      <Container sx={{ mt: 4 }}>
        <Alert
          severity="error"
          sx={{
            mb: 2,
            borderRadius: 3,
            alignItems: 'center',
          }}
        >
          {error}
        </Alert>
        <Button
          variant="contained"
          onClick={onBackClick}
          startIcon={<ArrowBackIosNewIcon sx={{ fontSize: 16 }} />}
          sx={{
            borderRadius: 3,
            px: 3,
            py: 1,
            fontSize: 14,
            fontWeight: 500,
            textTransform: 'none',
            boxShadow: 'none',
            '&:active': {
              transform: 'scale(0.98)',
            },
          }}
          component={motion.div}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          Назад
        </Button>
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ py: 2 }}>
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
          px: 1,
        }}
      >
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            fontSize: 20,
          }}
        >
          История
        </Typography>
        <Button
          variant="text"
          onClick={onBackClick}
          startIcon={<ArrowBackIosNewIcon sx={{ fontSize: 16 }} />}
          sx={{
            borderRadius: 3,
            px: 2,
            py: 0.5,
            fontSize: 14,
            fontWeight: 500,
            textTransform: 'none',
            color: theme.palette.primary.main,
            '&:active': {
              backgroundColor: alpha(theme.palette.primary.main, 0.1),
            },
          }}
          component={motion.div}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          Назад
        </Button>
      </Box>

      {!historyItems || historyItems.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <Box
            sx={{
              textAlign: 'center',
              mt: 8,
              p: 4,
              borderRadius: 3,
              backgroundColor: theme.palette.background.paper,
            }}
          >
            <DescriptionIcon
              sx={{
                fontSize: 48,
                color: theme.palette.text.disabled,
                mb: 2,
              }}
            />
            <Typography variant="body1" color="text.secondary" sx={{ mb: 1 }}>
              История пуста
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ opacity: 0.7 }}
            >
              Здесь будут отображаться ваши анализы
            </Typography>
          </Box>
        </motion.div>
      ) : (
        <Grid container spacing={1}>
          {historyItems.map((item) => (
            <HistoryItem key={item._id} item={item} />
          ))}
        </Grid>
      )}
    </Container>
  );
}

export default HistorySection;
