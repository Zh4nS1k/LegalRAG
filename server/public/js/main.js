document.addEventListener('DOMContentLoaded', function() {
    // Initialize the app
    document.getElementById('emptyState').style.display = 'block';

    // Set up event listeners
    document.getElementById('uploadBtn').addEventListener('click', function() {
        document.getElementById('documentInput').click();
    });

    document.getElementById('documentInput').addEventListener('change', handleFileUpload);

    // Set up tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', switchTab);
    });

    // Configure marked.js
    marked.setOptions({
        breaks: true,
        gfm: true,
        highlight: function(code, lang) {
            if (hljs.getLanguage(lang)) {
                return hljs.highlight(lang, code).value;
            }
            return hljs.highlightAuto(code).value;
        }
    });
});

function handleFileUpload(e) {
    if (e.target.files.length > 0) {
        clearMessages();
        const file = e.target.files[0];

        // Validate file type
        if (!file.name.endsWith('.pdf')) {
            showError('Пожалуйста, загрузите файл в формате PDF');
            return;
        }

        document.getElementById('fileInfo').innerHTML = `
            <strong>Выбран файл:</strong> ${file.name} (${formatFileSize(file.size)})
        `;

        // Show loading state
        document.getElementById('loadingSection').style.display = 'block';

        // Upload the file
        uploadDocument(file);
    }
}

function clearMessages() {
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('errorState').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    document.getElementById('fileInfo').innerHTML = '';
}

function showError(message) {
    const errorElement = document.getElementById('errorState');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('document', file);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.text();
            throw new Error(error || `Server returned ${response.status}`);
        }

        const data = await response.json();

        // Hide loading state
        document.getElementById('loadingSection').style.display = 'none';

        if (!data) {
            showError('Не удалось получить данные анализа');
            return;
        }

        // Display the results
        displayResults(data);
    } catch (error) {
        console.error('Upload error:', error);
        document.getElementById('loadingSection').style.display = 'none';
        showError('Произошла ошибка при анализе документа. Пожалуйста, попробуйте позже.');
    }
}

function displayResults(data) {
    // Show result section
    const resultSection = document.getElementById('resultSection');
    resultSection.style.display = 'block';

    // Set document type
    document.getElementById('documentType').textContent = `Тип документа: ${data.document_type || 'Неизвестно'}`;

    // Clear previous content
    const fullContainer = document.getElementById('fullContainer');
    const risksContainer = document.getElementById('risksContainer');
    const recommendationsContainer = document.getElementById('recommendationsContainer');
    const summaryContainer = document.getElementById('summaryContainer');

    fullContainer.innerHTML = '';
    risksContainer.innerHTML = '';
    recommendationsContainer.innerHTML = '';
    summaryContainer.innerHTML = '';

    // Convert markdown to HTML
    if (data.analysis) {
        const htmlContent = marked.parse(data.analysis);
        fullContainer.innerHTML = htmlContent;

        // Split into sections
        const sections = splitAnalysisIntoSections(htmlContent);

        // Display in appropriate containers
        risksContainer.innerHTML = sections.risks || '<p>Риски не выявлены.</p>';
        recommendationsContainer.innerHTML = sections.recommendations || '<p>Рекомендации отсутствуют.</p>';
        summaryContainer.innerHTML = sections.summary || '<p>Нет данных для сводки.</p>';

        // Highlight code blocks
        document.querySelectorAll('pre code').forEach(block => {
            hljs.highlightElement(block);
        });
    } else {
        fullContainer.innerHTML = '<p>Анализ не содержит данных</p>';
    }
}

function splitAnalysisIntoSections(htmlContent) {
    const sections = {
        risks: '',
        recommendations: '',
        summary: ''
    };

    // Split by headings
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = htmlContent;

    // Get all heading elements
    const headings = tempDiv.querySelectorAll('h3, h4');
    let currentSection = null;

    headings.forEach(heading => {
        const text = heading.textContent.toLowerCase();

        if (text.includes('правовые риски') || text.includes('неясные формулировки') ||
            text.includes('возможные нарушения') || text.includes('риски')) {
            currentSection = 'risks';
        } else if (text.includes('рекомендации')) {
            currentSection = 'recommendations';
        } else if (text.includes('заключение') || text.includes('сводка') ||
            text.includes('итог') || text.includes('вывод')) {
            currentSection = 'summary';
        }

        if (currentSection) {
            // Get content until next heading
            let content = '';
            let nextElement = heading.nextElementSibling;

            while (nextElement && !['H3', 'H4'].includes(nextElement.tagName)) {
                content += nextElement.outerHTML;
                nextElement = nextElement.nextElementSibling;
            }

            sections[currentSection] += heading.outerHTML + content;
        }
    });

    return sections;
}

function switchTab(e) {
    // Update active tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    e.target.classList.add('active');

    // Update active content
    const tabName = e.target.dataset.tab;
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`${tabName}Tab`).classList.add('active');
}   