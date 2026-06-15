// Admin JavaScript

let currentPage = 1;
let currentCategory = '';
const pageSize = 20;

// Initialize admin interface
document.addEventListener('DOMContentLoaded', function () {
  loadCategories();
  loadDocuments();
  loadStats();

  // Event listeners
  document
    .getElementById('uploadForm')
    .addEventListener('submit', handleUpload);
  document
    .getElementById('searchForm')
    .addEventListener('submit', handleSearch);
  document
    .getElementById('refreshBtn')
    .addEventListener('click', loadDocuments);
  document
    .getElementById('filterCategory')
    .addEventListener('change', handleCategoryFilter);
  document
    .getElementById('prevPage')
    .addEventListener('click', () => changePage(-1));
  document
    .getElementById('nextPage')
    .addEventListener('click', () => changePage(1));
});

// Load categories for dropdowns
async function loadCategories() {
  try {
    const response = await fetch('/api/admin/rag/categories', {
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    if (!response.ok) throw new Error('Failed to load categories');

    const data = await response.json();
    const categories = data.categories;

    // Populate category dropdowns
    populateCategoryDropdown('category', categories);
    populateCategoryDropdown('searchCategory', categories);
    populateCategoryDropdown('filterCategory', categories);
  } catch (error) {
    console.error('Error loading categories:', error);
    showError('Ошибка загрузки категорий');
  }
}

function populateCategoryDropdown(elementId, categories) {
  const select = document.getElementById(elementId);
  const currentValue = select.value;

  // Clear existing options except the first one
  while (select.children.length > 1) {
    select.removeChild(select.lastChild);
  }

  // Add new options
  categories.forEach((category) => {
    const option = document.createElement('option');
    option.value = category;
    option.textContent = category;
    select.appendChild(option);
  });

  // Restore previous value if it exists
  if (currentValue && categories.includes(currentValue)) {
    select.value = currentValue;
  }
}

// Handle document upload
async function handleUpload(event) {
  event.preventDefault();

  const formData = new FormData(event.target);
  const submitBtn = event.target.querySelector('button[type="submit"]');

  try {
    submitBtn.disabled = true;
    submitBtn.textContent = 'Загрузка...';

    const response = await fetch('/api/admin/rag/upload', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Upload failed');
    }

    const data = await response.json();
    showSuccess('Документ успешно загружен и поставлен в очередь на обработку');
    event.target.reset();

    // Refresh documents list
    setTimeout(loadDocuments, 1000);
  } catch (error) {
    console.error('Upload error:', error);
    showError(`Ошибка загрузки: ${error.message}`);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = 'Загрузить документ';
  }
}

// Handle document search
async function handleSearch(event) {
  event.preventDefault();

  const formData = new FormData(event.target);
  const searchData = {
    query: formData.get('query'),
    category: formData.get('category'),
    limit: parseInt(formData.get('limit')) || 10,
  };

  try {
    const response = await fetch('/api/admin/rag/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${getToken()}`,
      },
      body: JSON.stringify(searchData),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Search failed');
    }

    const data = await response.json();
    displaySearchResults(data.results);
  } catch (error) {
    console.error('Search error:', error);
    showError(`Ошибка поиска: ${error.message}`);
  }
}

// Display search results
function displaySearchResults(results) {
  const container = document.getElementById('searchResults');

  if (results.length === 0) {
    container.innerHTML = '<div class="loading">Результаты не найдены</div>';
    return;
  }

  const html = results
    .map(
      (result) => `
        <div class="search-result">
            <h4>${result.title}</h4>
            <div class="document-meta">
                <span>Категория: ${result.category}</span>
                <span>Источник: ${result.source || 'Не указан'}</span>
                <span class="similarity">Схожесть: ${(
                  result.similarity * 100
                ).toFixed(1)}%</span>
            </div>
            <div class="chunk-content">${result.chunk_content}</div>
        </div>
    `
    )
    .join('');

  container.innerHTML = html;
}

// Load documents list
async function loadDocuments() {
  const container = document.getElementById('documentsList');
  container.innerHTML = '<div class="loading">Загрузка документов...</div>';

  try {
    const offset = (currentPage - 1) * pageSize;
    const url = `/api/admin/rag/documents?limit=${pageSize}&offset=${offset}${
      currentCategory ? `&category=${currentCategory}` : ''
    }`;

    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to load documents');
    }

    const data = await response.json();
    displayDocuments(data.documents);
    updatePagination(data.count);
  } catch (error) {
    console.error('Error loading documents:', error);
    container.innerHTML = '<div class="error">Ошибка загрузки документов</div>';
  }
}

// Display documents
function displayDocuments(documents) {
  const container = document.getElementById('documentsList');

  if (documents.length === 0) {
    container.innerHTML = '<div class="loading">Документы не найдены</div>';
    return;
  }

  const html = documents
    .map(
      (doc) => `
        <div class="document-card">
            <h3>${doc.title}</h3>
            <div class="document-meta">
                <span>Категория: ${doc.category}</span>
                <span>Источник: ${doc.source || 'Не указан'}</span>
                <span>Файл: ${doc.filename}</span>
                <span>Загружен: ${new Date(
                  doc.created_at
                ).toLocaleDateString()}</span>
                <span class="document-status ${doc.status}">${getStatusText(
        doc.status
      )}</span>
            </div>
            <div class="document-actions">
                <button class="reprocess-btn" onclick="reprocessDocument('${
                  doc.id
                }')" ${doc.status === 'processing' ? 'disabled' : ''}>
                    Переобработать
                </button>
                <button class="delete-btn" onclick="deleteDocument('${
                  doc.id
                }')">
                    Удалить
                </button>
            </div>
        </div>
    `
    )
    .join('');

  container.innerHTML = html;
}

// Get status text
function getStatusText(status) {
  const statusMap = {
    pending: 'Ожидает обработки',
    processing: 'Обрабатывается',
    processed: 'Обработан',
    error: 'Ошибка',
  };
  return statusMap[status] || status;
}

// Update pagination
function updatePagination(totalCount) {
  const totalPages = Math.ceil(totalCount / pageSize);
  const pageInfo = document.getElementById('pageInfo');
  const prevBtn = document.getElementById('prevPage');
  const nextBtn = document.getElementById('nextPage');

  pageInfo.textContent = `Страница ${currentPage} из ${totalPages}`;
  prevBtn.disabled = currentPage <= 1;
  nextBtn.disabled = currentPage >= totalPages;
}

// Change page
function changePage(delta) {
  currentPage += delta;
  if (currentPage < 1) currentPage = 1;
  loadDocuments();
}

// Handle category filter
function handleCategoryFilter(event) {
  currentCategory = event.target.value;
  currentPage = 1;
  loadDocuments();
}

// Load statistics
async function loadStats() {
  const container = document.getElementById('statsContainer');
  container.innerHTML = '<div class="loading">Загрузка статистики...</div>';

  try {
    const response = await fetch('/api/admin/rag/stats', {
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to load stats');
    }

    const data = await response.json();
    displayStats(data.stats);
  } catch (error) {
    console.error('Error loading stats:', error);
    container.innerHTML = '<div class="error">Ошибка загрузки статистики</div>';
  }
}

// Display statistics
function displayStats(stats) {
  const container = document.getElementById('statsContainer');

  let html = `
        <div class="stat-card">
            <h3>${stats.total}</h3>
            <p>Всего документов</p>
        </div>
    `;

  // Status statistics
  if (stats.status_stats && stats.status_stats.length > 0) {
    html += '<div class="status-stats">';
    html += '<h4>По статусам</h4>';
    stats.status_stats.forEach((stat) => {
      html += `
                <div class="stat-item">
                    <span>${getStatusText(stat._id)}</span>
                    <span>${stat.count}</span>
                </div>
            `;
    });
    html += '</div>';
  }

  // Category statistics
  if (stats.category_stats && stats.category_stats.length > 0) {
    html += '<div class="category-stats">';
    html += '<h4>По категориям</h4>';
    stats.category_stats.forEach((stat) => {
      html += `
                <div class="stat-item">
                    <span>${stat._id}</span>
                    <span>${stat.count}</span>
                </div>
            `;
    });
    html += '</div>';
  }

  container.innerHTML = html;
}

// Delete document
async function deleteDocument(docId) {
  if (!confirm('Вы уверены, что хотите удалить этот документ?')) {
    return;
  }

  try {
    const response = await fetch(`/api/admin/rag/documents/${docId}`, {
      method: 'DELETE',
      headers: {
        Authorization: `Bearer ${getToken()}`,
      },
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Delete failed');
    }

    showSuccess('Документ успешно удален');
    loadDocuments();
    loadStats();
  } catch (error) {
    console.error('Delete error:', error);
    showError(`Ошибка удаления: ${error.message}`);
  }
}

// Reprocess document
async function reprocessDocument(docId) {
  try {
    const response = await fetch(
      `/api/admin/rag/documents/${docId}/reprocess`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${getToken()}`,
        },
      }
    );

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Reprocess failed');
    }

    showSuccess('Документ поставлен в очередь на переобработку');
    setTimeout(loadDocuments, 1000);
  } catch (error) {
    console.error('Reprocess error:', error);
    showError(`Ошибка переобработки: ${error.message}`);
  }
}

// Utility functions
function getToken() {
  // Get token from localStorage or wherever you store it
  return localStorage.getItem('accessToken') || '';
}

function showSuccess(message) {
  const div = document.createElement('div');
  div.className = 'success';
  div.textContent = message;
  document.body.insertBefore(div, document.body.firstChild);

  setTimeout(() => {
    div.remove();
  }, 5000);
}

function showError(message) {
  const div = document.createElement('div');
  div.className = 'error';
  div.textContent = message;
  document.body.insertBefore(div, document.body.firstChild);

  setTimeout(() => {
    div.remove();
  }, 5000);
}
