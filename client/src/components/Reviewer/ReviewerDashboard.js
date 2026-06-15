import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import evaluationService from '../../services/evaluationService';
import { Container, Card, Box, Typography } from '@mui/material';

const ReviewerDashboard = () => {
    const [tasks, setTasks] = useState([]);
    const [loading, setLoading] = useState(true);
    const navigate = useNavigate();

    useEffect(() => {
        loadMyTasks();
    }, []);

    const loadMyTasks = async () => {
        try {
            const response = await evaluationService.getMyTasks();
            setTasks(response.data || []);
        } catch (err) {
            console.error('Failed to load tasks', err);
            setTasks([]);
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <div>Loading assigned tasks...</div>;

    return (
        <Container maxWidth="lg" sx={{ py: 4 }} className="reviewer-dashboard">
            <Card sx={{ p: 4, mb: 4, borderRadius: '16px', boxShadow: 'none', border: '1px solid #E5E7EB' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h4" fontWeight={600} gutterBottom>
                        Expert Evaluation Dashboard
                    </Typography>
                    <div className="stats-mini">
                        <span className="badge status-pending">{tasks.filter(t => t.status === 'Pending').length} Pending</span>
                    </div>
                </Box>
            </Card>

            <div className="task-list">
                {(!tasks || tasks.length === 0) ? (
                    <div className="empty-state">
                        <p>No pending evaluations assigned to you at this time.</p>
                    </div>
                ) : (
                    <div className="task-grid">
                        {tasks.map(task => (
                            <div key={task.id} className="task-card" onClick={() => navigate(`/reviewer/eval/${task.id}`, { state: { task } })}>
                                <div className="task-card-header">
                                    <span className="task-id">#{task.external_id || task.id.substring(0, 8)}</span>
                                    <span className={`badge status-${task.status.toLowerCase()}`}>{task.status}</span>
                                </div>
                                <h3>{task.question}</h3>
                                <div className="task-footer">
                                    <span className="date">Assigned: {new Date(task.created_at).toLocaleDateString()}</span>
                                    <button className="eval-link">Evaluate &rarr;</button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </Container>
    );
};

export default ReviewerDashboard;
