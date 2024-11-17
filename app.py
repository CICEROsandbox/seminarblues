import { useState } from 'react';
import { Search, Tag, ChevronDown, ChevronUp } from 'lucide-react';

const SeminarRecommendations = () => {
  const [description, setDescription] = useState('');
  const [keywords, setKeywords] = useState([]);
  const [selectedKeywords, setSelectedKeywords] = useState(new Set());
  const [results, setResults] = useState([]);
  const [expandedResults, setExpandedResults] = useState(new Set());

  // Simulate keyword extraction
  const generateKeywords = (text) => {
    const sampleKeywords = text.toLowerCase()
      .split(/\s+/)
      .filter(word => word.length > 3)
      .map(word => ({
        text: word,
        selected: true
      }));
    setKeywords(sampleKeywords);
    setSelectedKeywords(new Set(sampleKeywords.map(k => k.text)));
  };

  // Toggle keyword selection
  const toggleKeyword = (keyword) => {
    const newSelected = new Set(selectedKeywords);
    if (newSelected.has(keyword)) {
      newSelected.delete(keyword);
    } else {
      newSelected.add(keyword);
    }
    setSelectedKeywords(newSelected);
  };

  // Simulate search functionality
  const searchSpeakers = () => {
    // Mock results
    const mockResults = [
      {
        id: 1,
        name: "Dr. Anna Hansen",
        relevance: 0.89,
        source: "Climate Conference 2023",
        content: "Led a discussion on renewable energy transitions and climate adaptation strategies in urban environments."
      },
      {
        id: 2,
        name: "Prof. Erik Larsson",
        relevance: 0.78,
        source: "Environmental Summit",
        content: "Presented research on sustainable urban development and green infrastructure implementation."
      }
    ];
    setResults(mockResults);
    setExpandedResults(new Set([1])); // Auto-expand first result
  };

  // Highlight keywords in text
  const highlightKeywords = (text) => {
    if (!text) return '';
    const parts = text.split(new RegExp(`(${Array.from(selectedKeywords).join('|')})`, 'gi'));
    return parts.map((part, i) => 
      selectedKeywords.has(part.toLowerCase()) ? 
        <span key={i} className="bg-yellow-200 px-1 rounded">{part}</span> : part
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Description Input */}
      <div className="space-y-2">
        <label className="block text-sm font-medium">Beskriv seminar-temaet:</label>
        <textarea
          className="w-full h-32 p-3 border rounded-lg shadow-sm"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Eksempel: Et seminar om klimatilpasning og hetebølger, med fokus på helsekonsekvenser for eldre."
        />
        <button
          onClick={() => generateKeywords(description)}
          className="mt-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
        >
          <Tag size={16} />
          Generer nøkkelord
        </button>
      </div>

      {/* Keywords Selection */}
      {keywords.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium">Velg relevante nøkkelord:</h3>
          <div className="flex flex-wrap gap-2">
            {keywords.map((keyword, index) => (
              <button
                key={index}
                onClick={() => toggleKeyword(keyword.text)}
                className={`px-3 py-1 rounded-full text-sm font-medium transition-colors
                  ${selectedKeywords.has(keyword.text) 
                    ? 'bg-blue-100 text-blue-800 hover:bg-blue-200' 
                    : 'bg-red-100 text-red-800 hover:bg-red-200'}`}
              >
                {keyword.text}
              </button>
            ))}
          </div>
          <button
            onClick={searchSpeakers}
            className="mt-4 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center gap-2"
          >
            <Search size={16} />
            Søk etter deltakere
          </button>
        </div>
      )}

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-xl font-semibold">Foreslåtte deltakere</h2>
          {results.map((result) => (
            <div key={result.id} className="border rounded-lg shadow-sm">
              <button
                onClick={() => {
                  const newExpanded = new Set(expandedResults);
                  if (newExpanded.has(result.id)) {
                    newExpanded.delete(result.id);
                  } else {
                    newExpanded.add(result.id);
                  }
                  setExpandedResults(newExpanded);
                }}
                className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-50"
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium">{result.name}</span>
                  <span className="text-sm text-gray-500">
                    {(result.relevance * 100).toFixed(0)}% relevans
                  </span>
                </div>
                {expandedResults.has(result.id) ? 
                  <ChevronUp size={20} /> : 
                  <ChevronDown size={20} />
                }
              </button>
              {expandedResults.has(result.id) && (
                <div className="px-4 py-3 border-t">
                  <p className="text-sm text-gray-600 mb-2">
                    <strong>Kilde:</strong> {result.source}
                  </p>
                  <p className="text-sm">
                    {highlightKeywords(result.content)}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SeminarRecommendations;
