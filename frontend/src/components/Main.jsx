import React, { useEffect, useState } from 'react';

const lots = [
    { lot_id: "A" },
    { lot_id: "B" },
    { lot_id: "C" },
    { lot_id: "D" },
];

const Main = () => {
    const [selected, setSelected] = useState(0);
    const [carData, setCarData] = useState([]);
    const [unauthorizedCars, setUnauthorizedCars] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                const response = await fetch(`http://localhost:8000/current_cars/${lots[selected].lot_id}`, { method: "GET" });
                const data = await response.json();
                if (response.ok) {
                    setCarData(data.cars || []);

                    // Filter unauthorized cars
                    const unauthorized = (data.cars || []).filter(car => car.registered_lot_id !== lots[selected].lot_id);
                    setUnauthorizedCars(unauthorized);
                } else {
                    setError(data.detail || "Failed to fetch data.");
                    setCarData([]);
                    setUnauthorizedCars([]);
                }
            } catch (err) {
                setError(`${err}`);
                setCarData([]);
                setUnauthorizedCars([]);
            }
            setLoading(false);
        };

        fetchData();
    }, [selected]);

    return (
        <section id="main" className="h-full flex flex-col gap-2">
            {/* TOP CONTAINER */}
            <div className="w-full h-fit p-6 flex flex-row items-center gap-4">
                <div>
                    <h6 className="text-sm font-thin">Select Lot Type</h6>
                    <p className="text-sm font-bold">{lots.length} available lots</p>
                </div>
                {lots.map((lot, i) => (
                    <button
                        key={lot.lot_id}
                        className={`${
                            selected === i ? "border-reddish text-white bg-reddish" : "border-dark-gray"
                        } font-bold w-fit p-4 rounded-xl border-[1px]`}
                        onClick={() => setSelected(i)}
                    >
                        {lot.lot_id}
                    </button>
                ))}
            </div>
            {/* BOTTOM CONTAINER */}
            <div className="w-full h-[60%] p-6 flex justify-center items-start gap-8">
                {/* Authorized Car Data */}
                <div className="p-4 w-[60%] h-full rounded-xl border-[1px] border-dark-gray flex flex-col gap-4 overflow-y-scroll">
                    <h6 className="text-lg font-semibold">Cars in Lot {lots[selected].lot_id}</h6>

                    {loading ? (
                        <p className="text-gray-500">Loading...</p>
                    ) : error ? (
                        <p className="text-red-500">{error}</p>
                    ) : carData.length === 0 ? (
                        <p className="text-gray-500">No cars found in this lot.</p>
                    ) : (
                        <div className="overflow-x-auto">
                            {/* Table */}
                            <table className="min-w-full border-collapse border border-gray-300">
                                <thead>
                                    <tr className="bg-gray-200 text-gray-700 text-left">
                                        <th className="border border-gray-300 px-4 py-2">Plate Number</th>
                                        <th className="border border-gray-300 px-4 py-2">Registered Lot</th>
                                        <th className="border border-gray-300 px-4 py-2">Owner Name</th>
                                        <th className="border border-gray-300 px-4 py-2">Email</th>
                                        <th className="border border-gray-300 px-4 py-2">Entry Time</th>
                                    </tr>
                                </thead>
                                <tbody className='text-sm'>
                                    {carData.map((car, index) => (
                                        <tr key={index} className="border border-gray-300">
                                            <td className="border border-gray-300 px-4 py-2">{car.car_plate_num}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.registered_lot_id || "N/A"}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.owner_name || "Unknown"}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.registered_email || "Unknown"}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.enter_time}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>

                {/* Unauthorized Vehicles */}
                <div className="p-4 w-[40%] h-full rounded-xl border-[1px] border-dark-gray flex flex-col gap-4 overflow-y-scroll">
                    <h6 className="text-lg font-semibold text-red-600">
                        Unauthorized Vehicles ({unauthorizedCars.length})
                    </h6>

                    {unauthorizedCars.length === 0 ? (
                        <p className="text-gray-500">No unauthorized vehicles.</p>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="min-w-full border-collapse border border-gray-300">
                                <thead>
                                    <tr className="bg-red-200 text-red-700 text-left">
                                        <th className="border border-gray-300 px-4 py-2">Plate Number</th>
                                        <th className="border border-gray-300 px-4 py-2">Registered Lot</th>
                                        <th className="border border-gray-300 px-4 py-2">Owner Name</th>
                                        <th className="border border-gray-300 px-4 py-2">Email</th>
                                    </tr>
                                </thead>
                                <tbody className='text-sm'>
                                    {unauthorizedCars.map((car, index) => (
                                        <tr key={index} className="border border-gray-300 bg-red-100">
                                            <td className="border border-gray-300 px-4 py-2">{car.car_plate_num}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.registered_lot_id || "N/A"}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.owner_name || "Unknown"}</td>
                                            <td className="border border-gray-300 px-4 py-2">{car.registered_email || "Unknown"}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
};

export default Main;
