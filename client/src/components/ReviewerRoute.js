import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';

const ReviewerRoute = ({ children, userData }) => {
    const location = useLocation();
    const token = localStorage.getItem('token');

    if (!token) {
        return <Navigate to="/login" state={{ from: location }} replace />;
    }

    // Allowed roles for Reviewer dashboards: admin (for testing), professor, student
    const allowedRoles = ['admin', 'professor', 'student'];

    if (userData && !allowedRoles.includes(userData.role)) {
        return <Navigate to="/unauthorized" replace />;
    }

    return children;
};

export default ReviewerRoute;
