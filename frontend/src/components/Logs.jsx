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
            console.log(attempts)
            try {
                const response = await fetch(`${process.env.REACT_APP_BACKEND_URL}/flagged_cars/`);
                if (!response.ok) {
                throw new Error('Failed to fetch logs');
                }
                const data = await response.json();
                setLogs(data.cars.sort((a, b) => new Date(b.flag_time) - new Date(a.flag_time)));
                setSuspiciousLogs(data.suspicious_cars || []);
                success = true;
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

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <section className='w-full mt-12 p-[24px]'>
      <div className='flex gap-8'>
        {/* LEFT CONTAINER: Flagged Cars */}
        <div className='rounded-xl w-full h-[60vh] overflow-y-scroll p-2 border-[1px] border-dark-gray flex-1'>
          <div className='flex flex-col gap-4 p-[24px]'>
            <h6 className='font-bold'>Flagged Car Logs</h6>
            <table className="table-auto w-full border-collapse">
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
                  <tr className="bg-red-200 text-gray-700 text-left">
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
