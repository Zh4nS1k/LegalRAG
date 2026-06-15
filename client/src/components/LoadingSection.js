import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Fade,
  styled,
  Button,
  Alert,
  keyframes,
} from '@mui/material';

// Анимации
const slidePaper = keyframes`
  0% { top: -100%; }
  15% { top: -5%; }
  85% { top: -5%; }
  100% { top: 110%; }
`;

const scanLineAnim = keyframes`
  0% { top: 0%; opacity: 0.6; }
  50% { opacity: 1; }
  100% { top: 100%; opacity: 0.3; }
`;

// Стили
const LoadingContainer = styled(Box)(() => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '40px',
  background: '#ffffff',
  borderRadius: '16px',
  boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
  margin: '40px auto',
  maxWidth: '900px',
  textAlign: 'center',
}));

const PaperScannerContainer = styled(Box)(() => ({
  position: 'relative',
  width: '260px',
  height: '360px',
  backgroundColor: '#e6e6e6',
  border: '2px solid #ccc',
  borderRadius: '12px',
  overflow: 'hidden',
  boxShadow: 'inset 0 0 12px rgba(0,0,0,0.15)',
  margin: '24px auto',
}));

const Paper = styled(Box)(() => ({
  position: 'absolute',
  top: '-100%',
  left: '10%',
  width: '80%',
  height: '100%',
  backgroundColor: '#fff',
  borderRadius: '6px',
  padding: '18px 16px',
  fontSize: '11px',
  fontFamily: '"Georgia", serif',
  lineHeight: 1.45,
  color: '#333',
  boxShadow: '0 2px 6px rgba(0,0,0,0.1)',
  animation: `${slidePaper} 7s ease-in-out infinite`,
  zIndex: 2,
}));

const ScanningLine = styled(Box)(() => ({
  position: 'absolute',
  top: 0,
  left: 0,
  width: '100%',
  height: '5px',
  background: 'linear-gradient(to right, transparent, #E60000, transparent)',
  boxShadow: '0 0 15px rgba(230, 0, 0, 0.8)',
  animation: `${scanLineAnim} 2.5s linear infinite`,
  zIndex: 3,
}));

const Stamp = styled(Box)(() => ({
  position: 'absolute',
  bottom: '40px',
  right: '16px',
  width: '80px',
  height: '80px',
  borderRadius: '50%',
  backgroundColor: 'rgba(230, 0, 0, 0.05)',
  border: '2px dashed #E60000',
  transform: 'rotate(-10deg)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  fontSize: '8px',
  color: '#E60000',
  fontWeight: 800,
  textAlign: 'center',
  textTransform: 'uppercase',
  letterSpacing: '0.7px',
  zIndex: 4,
}));

const loadingMessages = [
  'Анализ структуры документа...',
  'Проверка соответствия ГК РК...',
  'Выявление налоговых рисков...',
  'Анализ договорных условий...',
  'Проверка на соответствие ТК РК...',
  'Поиск скрытых юридических рисков...',
  'Проверка реквизитов и подписей...',
  'Анализ на соответствие Налоговому кодексу РК...',
  'Оценка корпоративного права...',
  'Проверка соблюдения требований МФ РК...',
  'Анализ арбитражной практики...',
  'Сравнение с изменениями в законодательстве...',
  'Проверка лицензионных соглашений...',
  'Оценка договорных обязательств...',
  'Выявление конфликта интересов...',
  'Проверка на соответствие Закону "О бухгалтерском учете"...',
  'Анализ корпоративного управления...',
  'Проверка соблюдения AML-требований...',
  'Оценка налоговой оптимизации...',
  'Проверка на соответствие Закону "О защите персональных данных"...',
  'Анализ судебных перспектив...',
  'Проверка на наличие коррупционных рисков...',
  'Оценка договорной ответственности...',
  'Анализ финансовых последствий...',
  'Проверка на соответствие международным стандартам...',
];

function LoadingSection({ onCancel }) {
  const [message, setMessage] = useState(loadingMessages[0]);
  const [error] = useState(null);

  useEffect(() => {
    const interval = setInterval(() => {
      const index = Math.floor(Math.random() * loadingMessages.length);
      setMessage(loadingMessages[index]);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Fade in timeout={500}>
      <LoadingContainer>
        <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
          {message}
        </Typography>
        <PaperScannerContainer>
          <Paper>
            <Typography variant="subtitle2" fontWeight="bold" sx={{ mb: 1 }}>
              ДОГОВОР №001/2025
            </Typography>
            <Typography variant="body2" component="div">
              <p>г. Астана, 20 июня 2025 года</p>
              <p>
                Настоящий договор заключён между ТОО «ЮрЭксперт» и АО «КазЛегал»
                о предоставлении юридических услуг.
              </p>
              <ol style={{ paddingLeft: '1.2em', margin: 0 }}>
                <li>Срок действия договора: 12 месяцев.</li>
                <li>Стоимость услуг: 1 500 000 тенге.</li>
                <li>Ответственность сторон: регулируется ст. 351 ГК РК.</li>
              </ol>
              <p>Подписи сторон: _______________________</p>
            </Typography>
            <Stamp>
              LEGALLY.KZ
              <br />
              Проверено
            </Stamp>
          </Paper>
          <ScanningLine />
        </PaperScannerContainer>
        <Button
          variant="outlined"
          onClick={onCancel}
          sx={{
            mt: 2,
            textTransform: 'none',
            fontWeight: 600,
            color: '#333333',
            borderColor: '#E5E7EB',
            '&:hover': {
              borderColor: '#E60000',
              color: '#E60000',
              bgcolor: 'rgba(230,0,0,0.02)'
            }
          }}
        >
          Отменить анализ
        </Button>
        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}
      </LoadingContainer>
    </Fade>
  );
}

export default LoadingSection;
