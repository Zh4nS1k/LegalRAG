import React, { useState } from 'react';

const TaskDetailModal = ({ task, onClose, onUpdate }) => {
    const [editMode, setEditMode] = useState(false);
    const [formData, setFormData] = useState({
        question: task.question,
        answer: task.answer,
        chunks: task.chunks?.join('\n') || '',
        articles: task.articles?.join('\n') || ''
    });

    const isCompleted = task.status === 'COMPLETED';

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        const updatedTask = {
            ...formData,
            chunks: formData.chunks.split('\n').filter(c => c.trim()),
            articles: formData.articles.split('\n').filter(a => a.trim())
        };
        onUpdate(task.id, updatedTask);
    };

    return (
        <div className="modal-overlay">
            <div className="task-detail-modal">
                <div className="modal-header">
                    <h2>Task Details ({task.external_id || 'Manual'})</h2>
                    <span className={`status-badge status-${task.status?.toLowerCase()}`}>{task.status}</span>
                    <button className="close-btn" onClick={onClose}>&times;</button>
                </div>

                <div className="modal-body">
                    <form onSubmit={handleSubmit}>
                        <div className="form-group">
                            <label>Question</label>
                            <textarea
                                name="question"
                                value={formData.question}
                                onChange={handleChange}
                                readOnly={isCompleted || !editMode}
                                rows={3}
                            />
                        </div>

                        <div className="form-group">
                            <label>AI Answer</label>
                            <textarea
                                name="answer"
                                value={formData.answer}
                                onChange={handleChange}
                                readOnly={isCompleted || !editMode}
                                rows={6}
                            />
                        </div>

                        <div className="grid-2col">
                            <div className="form-group">
                                <label>Source Chunks (one per line)</label>
                                <textarea
                                    name="chunks"
                                    value={formData.chunks}
                                    onChange={handleChange}
                                    readOnly={isCompleted || !editMode}
                                    rows={8}
                                />
                            </div>
                            <div className="form-group">
                                <label>Legal Articles (one per line)</label>
                                <textarea
                                    name="articles"
                                    value={formData.articles}
                                    onChange={handleChange}
                                    readOnly={isCompleted || !editMode}
                                    rows={8}
                                />
                            </div>
                        </div>

                        {!isCompleted && (
                            <div className="modal-footer">
                                {!editMode ? (
                                    <button type="button" className="edit-btn" onClick={() => setEditMode(true)}>Edit Task</button>
                                ) : (
                                    <>
                                        <button type="submit" className="save-btn">Save Changes</button>
                                        <button type="button" className="cancel-btn" onClick={() => setEditMode(false)}>Cancel</button>
                                    </>
                                )}
                            </div>
                        )}
                    </form>

                    {isCompleted && task.result && (
                        <div className="evaluation-inspection">
                            <h3>Expert Evaluation Summary</h3>
                            <div className="results-grid">
                                <div className="result-card">
                                    <h4>Answer</h4>
                                    <div className="score">{task.result.answer_rating}/10</div>
                                    <p className="comment">{task.result.answer_comment}</p>
                                </div>
                                <div className="result-card">
                                    <h4>Chunks</h4>
                                    <div className="score">{task.result.chunks_rating}/10</div>
                                    <p className="comment">{task.result.chunks_comment}</p>
                                </div>
                                <div className="result-card">
                                    <h4>Articles</h4>
                                    <div className="score">{task.result.articles_rating}/10</div>
                                    <p className="comment">{task.result.articles_comment}</p>
                                </div>
                            </div>
                            <div className={`decision-banner ${task.result.confirm_action ? 'keep' : 'change'}`}>
                                Final Decision: {task.result.confirm_action ? 'KEEP AS IS' : 'NEEDS CHANGES'}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default TaskDetailModal;
