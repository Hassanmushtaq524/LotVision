import React, { useState, useEffect } from 'react';

const Logs = () => {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchLogs = async () => {
      try {
        const response = await fetch('http://localhost:8000/flagged_cars/');
        if (!response.ok) {
          throw new Error('Failed to fetch logs');
        }
        const data = await response.json();
        setLogs(data.cars);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
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
        {/* RIGHT CONTAINER */}
        {/* LEFT CONTAINER */}
        <div className='rounded-xl size-fit p-2 border-[1px] border-dark-gray'>
            <div className='flex flex-col gap-4 p-[24px]'>
            <h6 className='font-bold'>Flagged Car Logs</h6>
            <table className="table-auto w-fit border-collapse  overflow-y-scroll">
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
    </section>
  );
};

export default Logs;
