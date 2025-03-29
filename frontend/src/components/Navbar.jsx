import React from 'react'

const Navbar = ({ selectedPage, setSelectedPage }) => {
  return (
    <div id="navbar" className='w-full p-[24px] flex justify-between'>
      <h1 className='text-reddish italic'>LOT<span className='font-light'>VISION</span></h1>
      <div className='flex space-x-4'>
        <button
          className={`${
            selectedPage === "logs" ? "border-reddish text-white bg-reddish" : "border-dark-gray"
        } font-bold w-fit p-4 rounded-xl border-[1px]`}
          onClick={() => setSelectedPage('logs')}
        >
          Check Logs
        </button>
        <button
          className={`${
            selectedPage === "lots" ? "border-reddish text-white bg-reddish" : "border-dark-gray"
        } font-bold w-fit p-4 rounded-xl border-[1px]`}
          onClick={() => setSelectedPage('lots')}
        >
          View Lots
        </button>
      </div>
    </div>
  )
}

export default Navbar
