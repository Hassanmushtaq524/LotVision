import React, { useState, useEffect } from 'react';

const Logs = () => {
  const [logs, setLogs] = useState([]);
  const [suspiciousLogs, setSuspiciousLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLogs = async () => {
      let attempts = 0;
      const maxAttempts = 10;
      let success = false;

      while (attempts < maxAttempts && !success) {
        try {
          console.log(`Attempt: ${attempts + 1}`);

          const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/flagged_cars/`);
          if (!response.ok) {
            throw new Error('Failed to fetch logs');
          }
          const data = await response.json();
          setLogs(data.cars.sort((a, b) => new Date(b.flag_time) - new Date(a.flag_time)));
          success = true;

          // After fetching logs, analyze suspicious cars
          analyzeSuspiciousCars(data.cars);
        } catch (err) {
          attempts++;
          if (attempts >= maxAttempts) {
            setError(err.message);
          }
        } finally {
          setLoading(false);
        }
      }
    };

    fetchLogs();
  }, []);




  const analyzeSuspiciousCars = async (carsData) => {
    try {
      const API_KEY = process.env.REACT_APP_GEMINI_KEY;
      const prompt = generatePrompt(carsData);
  
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] }),
        }
      );
  
      const geminiData = await response.json();
  
      let geminiText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text.trim();
      console.log("Raw Gemini Text:", geminiText);
  
      // ✅ Handle case where JSON is wrapped in ```json ```
      if (geminiText.startsWith("```json")) {
        geminiText = geminiText.replace(/```json\s*/, "").replace(/```$/, "").trim();
      }
  
      let parsedData = [];
      try {
        parsedData = JSON.parse(geminiText);
        console.log("Parsed Suspicious Vehicles:", parsedData);
      } catch (jsonError) {
        console.error("Error parsing Gemini response:", jsonError);
        parsedData = [];
      }
  
      setSuspiciousLogs(parsedData);
    } catch (error) {
      console.error("Error analyzing suspicious vehicles:", error);
      setSuspiciousLogs([]);
    }
  };
  
  

  // 🔥 Function to generate the prompt for Gemini
  const generatePrompt = (carsData) => {
    let prompt = `
      Analyze the following vehicles and flag any suspicious vehicles based on their timestamps, number of entries, and flag reasons.
      Return only the license plate number and a note describing why it is more suspicious than other entries. Sort based on how suspicious the vehicle is. Output only 5 entries.\n 
      Return in the format [{ "car_plate_num": "XXX123", "note": "Reason" }] for multiple cars. Do not output anything else, and the output must start with [ and end with ], and not include quotations.\n\n
    `;
    carsData.forEach(car => {
      prompt += `Car Plate Number: ${car.car_plate_num}, Flag Time: ${car.flag_time}, Flag Reason: ${car.flag_reason}\n`;
    });
    return prompt;
  };

  if (loading) {
    return <div className='w-full flex justify-center'>Loading...</div>;
  }

  if (error) {
    return <div className='w-full flex justify-center'>Error: {error}</div>;
  }

  return (
    <section className='w-full mt-12 p-[24px]'>
      <div className='flex gap-8'>
        {/* LEFT CONTAINER: Flagged Cars */}
        <div className='rounded-xl w-[60vw] h-[60vh] p-[24px] border-[1px] border-dark-gray flex flex-col gap-4'>
          <h6 className='font-bold'>Flagged Car Logs</h6>
            <div className='overflow-y-scroll'>
                <table className="table-auto border-collapse">
                <thead>
                    <tr className="bg-gray-200 text-gray-700 text-left">
                    <th className="border border-gray-300 px-4 py-2">Car Plate Number</th>
                    <th className="border border-gray-300 px-4 py-2">Detected Lot ID</th>
                    <th className="border border-gray-300 px-4 py-2">Registered Lot ID</th>
                    <th className="border border-gray-300 px-4 py-2">Flag Reason</th>
                    <th className="border border-gray-300 px-4 py-2">Flag Time</th>
                    </tr>
                </thead>
                <tbody className='text-sm'>
                    {logs.map((car, index) => (
                    <tr key={index}>
                        <td className="border border-gray-300 px-4 py-2">{car.car_plate_num}</td>
                        <td className="border border-gray-300 px-4 py-2">{car.detected_lot_id}</td>
                        <td className="border border-gray-300 px-4 py-2">{car.registered_lot_id}</td>
                        <td className="border border-gray-300 px-4 py-2">{car.flag_reason}</td>
                        <td className="border border-gray-300 px-4 py-2">{car.flag_time}</td>
                    </tr>
                    ))}
                </tbody>
                </table>
            </div>
        </div>

        {/* RIGHT CONTAINER: Suspicious Vehicles */}
        <div className='rounded-xl size-fit p-2 border-[1px] border-dark-gray flex-1'>
          <div className='flex flex-col gap-4 p-[24px]'>
            <h6 className='font-bold'>Suspicious Vehicle Logs</h6>
            {suspiciousLogs.length > 0 ? (
              <table className="table-auto w-full border-collapse overflow-y-scroll">
                <thead>
                  <tr className="bg-pink-200 text-pink-700 text-left">
                    <th className="border border-gray-300 px-4 py-2">Car Plate Number</th>
                    <th className="border border-gray-300 px-4 py-2">Suspicious Note</th>
                  </tr>
                </thead>
                <tbody className='text-sm'>
                  {suspiciousLogs.map((suspiciousCar, index) => (
                    <tr key={index}>
                      <td className="border border-gray-300 px-4 py-2">{suspiciousCar.car_plate_num}</td>
                      <td className="border border-gray-300 px-4 py-2">{suspiciousCar.note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div>No suspicious vehicles found.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Logs;
