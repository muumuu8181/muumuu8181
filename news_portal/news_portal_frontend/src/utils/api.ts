import axios from "axios";

// API Types
export interface Article {
  id: string;
  title: string;
  content: string;
  url_id: string;
  published_date: string;
  fetched_date: string;
  tags: string[];
  source: string;
}

export interface URL {
  id: string;
  name: string;
  url: string;
  tags: string[];
  active: boolean;
  created_at: string;
}

export interface Tag {
  id: string;
  name: string;
  category: string;
}

// Create axios instance
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_BACKEND_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
export const api = {
  // Articles
  getArticles: async (params?: { 
    tags?: string[],
    start_date?: string,
    end_date?: string,
    source?: string
  }) => {
    const response = await apiClient.get<Article[]>('/api/articles', { params });
    return response.data;
  },
  
  getArticle: async (id: string) => {
    const response = await apiClient.get<Article>(`/api/articles/${id}`);
    return response.data;
  },

  // URLs
  getUrls: async () => {
    const response = await apiClient.get<URL[]>('/api/urls');
    return response.data;
  },

  createUrl: async (data: Omit<URL, 'id' | 'created_at'>) => {
    const response = await apiClient.post<URL>('/api/urls', data);
    return response.data;
  },

  updateUrl: async (id: string, data: Partial<URL>) => {
    const response = await apiClient.put<URL>(`/api/urls/${id}`, data);
    return response.data;
  },

  deleteUrl: async (id: string) => {
    await apiClient.delete(`/api/urls/${id}`);
  },

  // Tags
  getTags: async () => {
    const response = await apiClient.get<Tag[]>('/api/tags');
    return response.data;
  },

  createTag: async (data: Omit<Tag, 'id'>) => {
    const response = await apiClient.post<Tag>('/api/tags', data);
    return response.data;
  },

  getUrlTags: async () => {
    const response = await apiClient.get<{ url_id: string; tags: string[] }[]>('/api/urls/tags');
    return response.data;
  },

  addUrlTags: async (urlId: string, tags: string[]) => {
    const response = await apiClient.post(`/api/urls/${urlId}/tags`, tags);
    return response.data;
  },

  // Configuration
  reloadConfig: async () => {
    const response = await apiClient.post('/api/config/reload');
    return response.data;
  },
};

export default api;
