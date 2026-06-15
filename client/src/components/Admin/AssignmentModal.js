import React, { useState } from 'react';

const AssignmentModal = ({ task, users, onAssign, onClose }) => {
    const [selectedUserId, setSelectedUserId] = useState('');

    const reviewers = users.filter(u => u.role === 'professor' || u.role === 'student');

    return (
        <div className="modal-overlay">
            <div className="assignment-modal">
                <h3>Assign Task to Expert</h3>
                <p className="task-preview"><strong>Task ID:</strong> {task.external_id}</p>
                <p className="task-preview"><strong>Question:</strong> {task.question.substring(0, 100)}...</p>

                <div className="form-group">
                    <label>Select Reviewer</label>
                    <select
                        value={selectedUserId}
                        onChange={(e) => setSelectedUserId(e.target.value)}
                        className="reviewer-select"
                    >
                        <option value="">Choose a professor or student...</option>
                        {reviewers.map(u => (
                            <option key={u.id} value={u.id}>
                                {u.email} ({u.role})
                            </option>
                        ))}
                    </select>
                </div>

                <div className="modal-actions">
                    <button
                        className="primary-btn"
                        onClick={() => onAssign(task.id, selectedUserId)}
                        disabled={!selectedUserId}
                    >
                        Assign Task
                    </button>
                    <button className="secondary-btn" onClick={onClose}>Cancel</button>
                </div>
            </div>
        </div>
    );
};

export default AssignmentModal;
