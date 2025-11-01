import React from 'react';
import { Tab } from '../types';

interface TabsProps {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  tabs: Tab[];
  children: React.ReactNode;
}

const Tabs: React.FC<TabsProps> = ({ activeTab, onTabChange, tabs, children }) => {
  return (
    <div className="w-full">
      <div className="flex flex-wrap border-b border-gray-200">
        {tabs.map((tabItem) => (
          <button
            key={tabItem}
            className={`py-3 px-4 text-sm font-medium focus:outline-none transition-colors duration-200 ${
              activeTab === tabItem
                ? 'border-b-4 border-blue-500 text-blue-600 bg-blue-50'
                : 'text-gray-600 hover:text-gray-800 hover:bg-gray-50'
            }`}
            onClick={() => onTabChange(tabItem)}
          >
            {tabItem}
          </button>
        ))}
      </div>
      <div className="p-6 bg-white rounded-b-lg shadow-md min-h-[60vh]">
        {children}
      </div>
    </div>
  );
};

export default Tabs;
