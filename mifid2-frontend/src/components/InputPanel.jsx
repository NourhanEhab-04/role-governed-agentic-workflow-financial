import { useState } from 'react'

export default function InputPanel({ onRun, isLoading }) {
  const [clientInput, setClientInput] = useState('')
  const [productInput, setProductInput] = useState('')

  const canRun = clientInput.trim().length > 0 && productInput.trim().length > 0

  return (
    <div className="w-96 min-w-80 bg-white border-r border-gray-200 p-6 flex flex-col gap-4 shrink-0">

      <h1 className="text-lg font-semibold text-gray-800">
        MiFID II Suitability Assessment
      </h1>

      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-gray-600">
          Client Profile
        </label>
        <textarea
          className="w-full h-40 p-3 text-sm border border-gray-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 placeholder-gray-300"
          placeholder="e.g. John is a 45-year-old conservative investor with basic knowledge of equities, a 5-year investment horizon, €50,000 liquid assets, €60,000 annual income, investing €10,000. He cannot afford a total loss."
          value={clientInput}
          onChange={e => setClientInput(e.target.value)}
          disabled={isLoading}
        />
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-gray-600">
          Product Description
        </label>
        <textarea
          className="w-full h-40 p-3 text-sm border border-gray-200 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 placeholder-gray-300"
          placeholder="e.g. A leveraged ETF tracking the S&P 500. Risk class 6, requires moderate knowledge, minimum 3-year horizon, has total loss potential. Not a complex product."
          value={productInput}
          onChange={e => setProductInput(e.target.value)}
          disabled={isLoading}
        />
      </div>

      <button
        onClick={() => onRun(clientInput, productInput)}
        disabled={!canRun || isLoading}
        className="w-full py-2.5 rounded-lg text-sm font-medium transition-colors
          disabled:bg-gray-100 disabled:text-gray-300 disabled:cursor-not-allowed
          enabled:bg-blue-600 enabled:text-white enabled:hover:bg-blue-700 enabled:cursor-pointer"
      >
        {isLoading ? 'Running pipeline...' : 'Run Assessment'}
      </button>

    </div>
  )
}
