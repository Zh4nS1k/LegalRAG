import React from 'react';
import { render, screen } from '@testing-library/react';
import ResultSection from '../components/ResultSection';
import '@testing-library/jest-dom';

// Mock helpers because splitAnalysisIntoSections uses DOMParser which might not be in JSDOM
jest.mock('../utils/helpers', () => ({
  splitAnalysisIntoSections: (html) => ({
    risks: html.includes('Risk') ? '<h2>Risks</h2><p>High Risk</p>' : '',
    recommendations: '',
    summary: '',
  }),
}));

describe('ResultSection Component', () => {
  const mockData = {
    analysis: '# Legal Analysis\n\nSome content with Article 437.',
    document_type: 'Contract',
  };

  test('renders basic info correctly', () => {
    render(<ResultSection data={mockData} onBackClick={() => {}} />);
    
    expect(screen.getByText(/Результаты юридического анализа/i)).toBeInTheDocument();
    expect(screen.getByText(/Contract/i)).toBeInTheDocument();
  });

  test('renders tabs correctly', () => {
    render(<ResultSection data={mockData} onBackClick={() => {}} />);
    
    expect(screen.getByText(/Полный анализ/i)).toBeInTheDocument();
    expect(screen.getByText(/Риски/i)).toBeInTheDocument();
    expect(screen.getByText(/Рекомендации/i)).toBeInTheDocument();
  });

  test('handles empty analysis gracefully', () => {
    render(<ResultSection data={{}} onBackClick={() => {}} />);
    expect(screen.getByText(/Неизвестно/i)).toBeInTheDocument();
  });
});
