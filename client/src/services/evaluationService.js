import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080/api';

const getAuthHeader = () => {
    const token = localStorage.getItem('token');
    return {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    };
};

const evaluationService = {
    // Advanced Task & Upload Endpoints
    uploadGenerate: (formData) => {
        return axios.post(`${API_URL}/admin/tasks/upload/generate`, formData, getAuthHeader());
    },
    uploadReady: (formData) => {
        return axios.post(`${API_URL}/admin/tasks/upload/ready`, formData, getAuthHeader());
    },
    getAllTasks: (page = 1, limit = 10, status = '') => {
        return axios.get(`${API_URL}/admin/tasks?page=${page}&limit=${limit}&status=${status}`, getAuthHeader());
    },
    getTaskDetails: (id) => {
        return axios.get(`${API_URL}/admin/tasks/${id}`, getAuthHeader());
    },
    updateTask: (id, data) => {
        return axios.put(`${API_URL}/admin/tasks/${id}`, data, getAuthHeader());
    },
    deleteTask: (id) => {
        return axios.delete(`${API_URL}/admin/tasks/${id}`, getAuthHeader());
    },
    assignTask: (taskId, userId) => {
        return axios.post(`${API_URL}/admin/tasks/assign`, { task_id: taskId, user_id: userId }, getAuthHeader());
    },

    // Legacy/Mapping remains for compatibility with existing UI if needed during transition
    getParsedTasks: () => {
        return axios.get(`${API_URL}/admin/eval/parsed`, getAuthHeader());
    },
    getRatedResults: () => {
        return axios.get(`${API_URL}/admin/eval/rated`, getAuthHeader());
    },
    adminUpdateUserRole: (userId, role) => {
        return axios.post(`${API_URL}/admin/users/role`, { user_id: userId, role }, getAuthHeader());
    },
    getUsers: () => {
        return axios.get(`${API_URL}/admin/users`, getAuthHeader());
    },
    createUser: (userData) => {
        return axios.post(`${API_URL}/admin/users`, userData, getAuthHeader());
    },
    deleteUser: (userId) => {
        return axios.delete(`${API_URL}/admin/users/${userId}`, getAuthHeader());
    },
    exportResults: (format = 'csv') => {
        const token = localStorage.getItem('token');
        window.open(`${API_URL}/admin/eval/export?format=${format}&token=${token}`, '_blank');
    },

    // Reviewer Endpoints
    getMyTasks: () => {
        return axios.get(`${API_URL}/eval/my-tasks`, getAuthHeader());
    },
    submitEvaluation: (evaluationData) => {
        return axios.post(`${API_URL}/eval/submit`, evaluationData, getAuthHeader());
    }
};

export default evaluationService;
