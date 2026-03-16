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
    
    // ResultSection renders both a span and a Typography (caption) with the same text in the Tab label.
    // Use getAllByText and check that at least one is present.
    expect(screen.getAllByText(/Полный анализ/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Риски/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Рекомендации/i).length).toBeGreaterThanOrEqual(1);
  });

  test('handles empty analysis gracefully', () => {
    render(<ResultSection data={{}} onBackClick={() => {}} />);
    expect(screen.getByText(/Неизвестно/i)).toBeInTheDocument();
  });
});
