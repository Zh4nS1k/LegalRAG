import React, { useState, useEffect, useCallback, useRef } from 'react';
import evaluationService from '../../services/evaluationService';
import UserManagement from './UserManagement';
import axios from 'axios';
import { Container, Card, Paper, Box, Typography } from '@mui/material';
import TaskDetailModal from './TaskDetailModal';
import AssignmentModal from './AssignmentModal';
// Here is the Evalution admin page
const EvaluationAdmin = () => {
    const [view, setView] = useState('tasks'); // 'tasks' or 'users'
    const [tasks, setTasks] = useState([]);
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(false);

    // Pagination & Filtering
    const [page, setPage] = useState(1);
    const [totalTasks, setTotalTasks] = useState(0);
    const [statusFilter, setStatusFilter] = useState('');
    const limit = 10;

    // UI State
    const [selectedTask, setSelectedTask] = useState(null);
    const [showAssignModal, setShowAssignModal] = useState(false);
    const [showDetailModal, setShowDetailModal] = useState(false);
    const [exportFormat, setExportFormat] = useState('csv');

    const fileGenerateRef = useRef(null);
    const fileReadyRef = useRef(null);

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const [tasksRes, usersRes] = await Promise.all([
                evaluationService.getAllTasks(page, limit, statusFilter),
                evaluationService.getUsers()
            ]);
            setTasks(tasksRes.data.tasks || []);
            setTotalTasks(tasksRes.data.total || 0);
            setUsers(usersRes.data || []);
        } catch (err) {
            console.error('Failed to load admin data', err);
        } finally {
            setLoading(false);
        }
    }, [page, statusFilter, limit]);

    useEffect(() => {
        if (view === 'tasks') {
            loadData();
        }
    }, [view, loadData]);

    const handleUploadGenerate = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            await evaluationService.uploadGenerate(formData);
            alert('Bulk AI generation started. Tasks will appear shortly.');
            loadData();
        } catch (err) {
            alert('Upload failed: ' + err.message);
        }
    };

    const handleUploadReady = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            await evaluationService.uploadReady(formData);
            alert('Pre-generated tasks imported successfully.');
            loadData();
        } catch (err) {
            alert('Import failed: ' + err.message);
        }
    };

    const handleAssign = async (taskId, userId) => {
        try {
            await evaluationService.assignTask(taskId, userId);
            setShowAssignModal(false);
            loadData();
        } catch (err) {
            alert('Assignment failed');
        }
    };

    const handleUpdateTask = async (taskId, data) => {
        try {
            await evaluationService.updateTask(taskId, data);
            setShowDetailModal(false);
            loadData();
        } catch (err) {
            alert('Update failed');
        }
    };

    const handleDeleteTask = async (id) => {
        if (!window.confirm('Are you sure you want to delete this task?')) return;
        try {
            await evaluationService.deleteTask(id);
            loadData();
        } catch (err) {
            alert('Delete failed');
        }
    };

    const getReviewerName = (id) => {
        const user = users.find(u => u.id === id);
        return user ? user.email.split('@')[0] : 'Unassigned';
    };

    return (
        <Container maxWidth="lg" sx={{ py: 4 }} className="eval-admin-container">
            <Card sx={{ p: 4, mb: 4, borderRadius: '16px', boxShadow: 'none', border: '1px solid #E5E7EB' }}>
                <Typography variant="h4" fontWeight={600} gutterBottom>
                    Legally Admin Panel
                </Typography>
                <p className="subtitle">HITL Pipeline & Task Management</p>
                <div className="header-actions">
                    <select value={exportFormat} onChange={(e) => setExportFormat(e.target.value)}>
                        <option value="csv">CSV</option>
                        <option value="excel">Excel</option>
                        <option value="json">JSON</option>
                    </select>
                    <button className="export-btn" onClick={() => evaluationService.exportResults(exportFormat)}>
                        Export Rated
                    </button>
                </div>
            </Card>

            <div className="admin-nav">
                <button className={view === 'tasks' ? 'active' : ''} onClick={() => setView('tasks')}>Tasks</button>
                <button className={view === 'users' ? 'active' : ''} onClick={() => setView('users')}>Users</button>
            </div>

            {loading && <div className="loading-spinner">Loading tasks...</div>}

            {view === 'tasks' ? (
                <>
                    <div className="upload-pipelines">
                        <div className="pipeline-card generate">
                            <h3>Pipeline 1: AI Generation</h3>
                            <p>Upload raw questions (Excel/JSON) for AI to answer.</p>
                            <input
                                type="file"
                                hidden
                                ref={fileGenerateRef}
                                onChange={handleUploadGenerate}
                            />
                            <button onClick={() => fileGenerateRef.current.click()}>Upload Questions</button>
                        </div>
                        <div className="pipeline-card ready">
                            <h3>Pipeline 2: Direct Import</h3>
                            <p>Import already answered Q&A pairs directly.</p>
                            <input
                                type="file"
                                hidden
                                ref={fileReadyRef}
                                onChange={handleUploadReady}
                            />
                            <button onClick={() => fileReadyRef.current.click()}>Import Ready Tasks</button>
                        </div>
                    </div>

                    <Paper sx={{ p: 0, borderRadius: '16px', overflow: 'hidden', boxShadow: 'none', border: '1px solid #E5E7EB' }}>
                        <div className="table-header">
                            <h2>All Q&A Tasks</h2>
                            <div className="table-filters">
                                <select
                                    className="status-filter"
                                    value={statusFilter}
                                    onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
                                >
                                    <option value="">All Statuses</option>
                                    <option value="PENDING_ASSIGNMENT">Pending Assignment</option>
                                    <option value="ASSIGNED">Assigned</option>
                                    <option value="COMPLETED">Completed</option>
                                </select>
                                <button className="add-task-btn" onClick={() => {
                                    setSelectedTask({ question: '', answer: '', chunks: [], articles: [], status: 'PENDING_ASSIGNMENT' });
                                    setShowDetailModal(true);
                                }}>+ Add Single Task</button>
                            </div>
                        </div>

                        <table>
                            <thead>
                                <tr>
                                    <th>Status</th>
                                    <th>Ext ID</th>
                                    <th>Question</th>
                                    <th>Assigned To</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tasks.map(task => (
                                    <tr key={task.id}>
                                        <td><span className={`status-pill ${task.status?.toLowerCase()}`}>{task.status}</span></td>
                                        <td>{task.external_id}</td>
                                        <td className="truncate-cell">{task.question}</td>
                                        <td>{getReviewerName(task.assigned_to_user_id)}</td>
                                        <td className="actions-cell">
                                            <button onClick={() => { setSelectedTask(task); setShowDetailModal(true); }}>View/Edit</button>
                                            <button onClick={() => { setSelectedTask(task); setShowAssignModal(true); }}>Assign</button>
                                            <button className="del-btn" onClick={() => handleDeleteTask(task.id)}>Delete</button>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </Paper>

                    <Paper sx={{ p: 3, mb: 4, borderRadius: '16px', boxShadow: 'none', border: '1px solid #E5E7EB' }}>
                        <div className="pagination">
                            <button
                                disabled={page === 1}
                                onClick={() => setPage(p => p - 1)}
                                className="page-btn"
                            >
                                Previous
                            </button>
                            <span className="page-info">
                                Page {page} of {Math.max(1, Math.ceil(totalTasks / limit))} ({totalTasks} total tasks)
                            </span>
                            <button
                                disabled={page >= Math.ceil(totalTasks / limit)}
                                onClick={() => setPage(p => p + 1)}
                                className="page-btn"
                            >
                                Next
                            </button>
                        </div>
                    </Paper>
                </>
            ) : (
                <UserManagement />
            )}

            {showDetailModal && selectedTask && (
                <TaskDetailModal
                    task={selectedTask}
                    onClose={() => setShowDetailModal(false)}
                    onUpdate={handleUpdateTask}
                />
            )}

            {showAssignModal && selectedTask && (
                <AssignmentModal
                    task={selectedTask}
                    users={users}
                    onAssign={handleAssign}
                    onClose={() => setShowAssignModal(false)}
                />
            )}
        </Container>
    );
};

export default EvaluationAdmin;
