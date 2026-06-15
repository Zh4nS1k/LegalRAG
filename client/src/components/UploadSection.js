import React, { useRef, useState } from 'react';
import {
  Container,
  Typography,
  Button,
  Alert,
  Fade,
  Stack,
  Box,
  CircularProgress,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  UploadFile as UploadFileIcon,
  History as HistoryIcon,
  Description as DescriptionIcon,
  Close as CloseIcon,
  CloudUpload as CloudUploadIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { formatFileSize } from '../utils/helpers';
import { useNavigate } from 'react-router-dom';
import banner from '../images/banner_legally.jpg';

const FileUploadArea = styled(Box, {
  shouldForwardProp: (prop) => prop !== '$dragactive',
})(({ theme, $dragactive }) => ({
  padding: theme.spacing($dragactive ? 4 : 3),
  border: `2px dashed ${$dragactive ? '#E60000' : '#E5E7EB'}`,
  borderRadius: '12px',
  backgroundColor: $dragactive
    ? 'rgba(230, 0, 0, 0.05)'
    : '#FFFFFF',
  transition: 'all 0.3s ease',
  textAlign: 'center',
  cursor: 'pointer',
  position: 'relative',
  '&:hover': {
    borderColor: '#E60000',
    backgroundColor: 'rgba(230, 0, 0, 0.02)',
  },
}));

const FileInfoBox = styled(Box)(({ theme }) => ({
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginTop: theme.spacing(2),
}));

function UploadSection({
  onFileUpload,
  fileInfo,
  error,
  onHistoryClick,
  isLoading = false,
  onClearError,
  onCancelUpload,
  onRemoveFile,
}) {
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);
  const [fileError, setFileError] = useState(null);
  const navigate = useNavigate();

  const validateFile = (file) => {
    if (!file) return 'Файл не выбран';
    if (file.size > 10 * 1024 * 1024) {
      return `Файл слишком большой (${formatFileSize(
        file.size
      )}). Максимум 10MB`;
    }
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      return 'Поддерживаются только PDF файлы';
    }
    return null;
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    const validationError = validateFile(file);
    if (validationError) {
      setFileError(validationError);
      onFileUpload(null);
      fileInputRef.current.value = '';
      return;
    }
    setFileError(null);
    onFileUpload(file);
    setDragActive(false);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(e.type === 'dragenter' || e.type === 'dragover');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    const validationError = validateFile(file);
    if (validationError) {
      setFileError(validationError);
      onFileUpload(null);
      return;
    }
    setFileError(null);
    onFileUpload(file);
  };

  const handleSelectFileClick = () => {
    if (!fileInfo && !isLoading) {
      fileInputRef.current.click();
    }
  };

  return (
    <Fade in timeout={600}>
      <Container maxWidth="md" sx={{ mt: 6, mb: 4 }}>
        <Paper elevation={0} sx={{ p: 4, borderRadius: '16px', border: '1px solid #E5E7EB', overflow: 'hidden' }}>
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <Box
              component="img"
              src={banner}
              alt="Banner"
              sx={{
                width: '80%',
                aspectRatio: '16 / 9',
                objectFit: 'cover',
                borderRadius: '16px',
              }}
            />
          </Box>
          <Typography variant="h4" gutterBottom sx={{ fontWeight: 800, color: '#000000', mt: 3 }}>
            Анализ юридических документов
          </Typography>
          <Box sx={{ width: 60, height: 4, background: '#E60000', mb: 4 }} />

          {(error || fileError) && (
            <Alert
              severity="error"
              sx={{ mt: 2, mb: 3 }}
              action={
                <IconButton
                  size="small"
                  color="inherit"
                  onClick={() => {
                    onClearError();
                    setFileError(null);
                  }}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              }
            >
              {error || fileError}
            </Alert>
          )}

          <FileUploadArea
            $dragactive={dragActive}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={handleSelectFileClick}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf"
              hidden
            />

            {isLoading ? (
              <Box
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                }}
              >
                <CircularProgress
                  size={60}
                  thickness={4}
                  sx={{ mb: 2, color: '#E60000' }}
                />
                <Typography variant="h6">Идет анализ документа...</Typography>
                <Button
                  variant="outlined"
                  sx={{
                    mt: 2,
                    color: '#E60000',
                    borderColor: '#E60000',
                    '&:hover': { borderColor: '#CC0000', bgcolor: 'rgba(230,0,0,0.05)' }
                  }}
                >
                  Отменить анализ
                </Button>
              </Box>
            ) : fileInfo ? (
              <>
                <DescriptionIcon
                  sx={{ fontSize: 60, color: '#E60000', mb: 1 }}
                />
                <FileInfoBox>
                  <Typography variant="h6" gutterBottom>
                    {fileInfo.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {formatFileSize(fileInfo.size)}
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<CloseIcon />}
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveFile();
                    }}
                    sx={{
                      mt: 2,
                      color: '#E60000',
                      borderColor: '#E60000',
                      '&:hover': { borderColor: '#CC0000', bgcolor: 'rgba(230,0,0,0.05)' }
                    }}
                  >
                    Удалить файл
                  </Button>
                </FileInfoBox>
              </>
            ) : (
              <>
                <CloudUploadIcon
                  sx={{ fontSize: 60, color: '#E60000', mb: 2 }}
                />
                <Typography variant="h6" gutterBottom>
                  {dragActive
                    ? 'Отпустите файл для загрузки'
                    : 'Перетащите файл сюда или нажмите для выбора'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Поддерживаются только PDF файлы (максимум 10MB)
                </Typography>
              </>
            )}
          </FileUploadArea>

          <Stack
            direction="row"
            spacing={2}
            justifyContent="center"
            sx={{ mt: 4 }}
          >
            <Tooltip title="Выберите PDF файл для анализа">
              <Button
                variant={fileInfo ? 'outlined' : 'contained'}
                size="large"
                startIcon={<UploadFileIcon />}
                onClick={() => fileInputRef.current.click()}
                disabled={isLoading}
                sx={{
                  backgroundColor: fileInfo ? 'transparent' : '#E60000',
                  color: fileInfo ? '#E60000' : '#FFFFFF',
                  borderColor: '#E60000',
                  '&:hover': {
                    backgroundColor: fileInfo ? 'rgba(230,0,0,0.05)' : '#CC0000',
                    borderColor: '#E60000'
                  }
                }}
              >
                {fileInfo ? 'Заменить файл' : 'Выбрать файл'}
              </Button>
            </Tooltip>

            <Button
              variant="outlined"
              size="large"
              startIcon={<HistoryIcon />}
              onClick={() => navigate('/history')}
              disabled={isLoading}
              sx={{
                color: '#333333',
                borderColor: '#333333',
                '&:hover': {
                  borderColor: '#000000',
                  bgcolor: 'rgba(0,0,0,0.05)'
                }
              }}
            >
              История проверок
            </Button>
          </Stack>

          <Box sx={{ mt: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              <InfoIcon
                fontSize="small"
                sx={{ mr: 0.5, verticalAlign: 'middle' }}
              />
              Система проверит ваш документ на соответствие законодательству РК
            </Typography>
          </Box>
        </Paper>
      </Container>
    </Fade >
  );
}

export default UploadSection;
