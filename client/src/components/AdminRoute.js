import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';

const AdminRoute = ({ children, userData }) => {
    const location = useLocation();
    const token = localStorage.getItem('token');

    if (!token) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    if (userData && userData.role !== 'admin') {
        return <Navigate to="/unauthorized" replace />;
    }

    return children;
};

export default AdminRoute;
