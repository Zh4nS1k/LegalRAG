import React, { useState, useEffect } from 'react';
import evaluationService from '../../services/evaluationService';

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [newUser, setNewUser] = useState({ email: '', password: '', role: 'student' });
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        loadUsers();
    }, []);

    const loadUsers = async () => {
        try {
            const response = await evaluationService.getUsers();
            setUsers(response.data || []);
        } catch (err) {
            console.error('Failed to load users', err);
        }
    };

    const handleCreateUser = async (e) => {
        e.preventDefault();
        setLoading(true);
        try {
            await evaluationService.createUser(newUser);
            setNewUser({ email: '', password: '', role: 'student' });
            loadUsers();
            alert('User created successfully');
        } catch (err) {
            alert('Failed to create user: ' + (err.response?.data?.error || err.message));
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteUser = async (userId) => {
        if (!window.confirm('Are you sure you want to delete this user?')) return;
        try {
            await evaluationService.deleteUser(userId);
            loadUsers();
            alert('User deleted successfully');
        } catch (err) {
            alert('Failed to delete user');
        }
    };

    const handleRoleChange = async (userId, newRole) => {
        try {
            await evaluationService.adminUpdateUserRole(userId, newRole);
            loadUsers();
        } catch (err) {
            alert('Failed to update role');
        }
    };

    return (
        <div className="user-management-panel">
            <section className="create-user-section">
                <h3>Create New expert/User</h3>
                <form onSubmit={handleCreateUser} className="create-user-form">
                    <input
                        type="email"
                        placeholder="Email"
                        value={newUser.email}
                        onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                        required
                    />
                    <input
                        type="password"
                        placeholder="Password (min 8 chars)"
                        value={newUser.password}
                        onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                        required
                    />
                    <select
                        value={newUser.role}
                        onChange={(e) => setNewUser({ ...newUser, role: e.target.value })}
                    >
                        <option value="student">Student</option>
                        <option value="professor">Professor</option>
                        <option value="admin">Admin</option>
                        <option value="user">User</option>
                    </select>
                    <button type="submit" disabled={loading} className="primary-btn">
                        {loading ? 'Creating...' : 'Create User'}
                    </button>
                </form>
            </section>

            <section className="users-list-section">
                <h3>System Users</h3>
                <div className="users-table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Email</th>
                                <th>Role</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {users.map(u => (
                                <tr key={u.id}>
                                    <td>{u.email}</td>
                                    <td>
                                        <select
                                            value={u.role}
                                            onChange={(e) => handleRoleChange(u.id, e.target.value)}
                                            className="inline-role-select"
                                        >
                                            <option value="admin">Admin</option>
                                            <option value="professor">Professor</option>
                                            <option value="student">Student</option>
                                            <option value="user">User</option>
                                        </select>
                                    </td>
                                    <td>
                                        <button
                                            onClick={() => handleDeleteUser(u.id)}
                                            className="delete-btn-sm"
                                        >
                                            Delete
                                        </button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </section>
        </div>
    );
};

export default UserManagement;
