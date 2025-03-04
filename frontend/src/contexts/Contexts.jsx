import { createContext, useState } from 'react';

// Create context for passing around global data. For example, user data.
export const GlobalContext = createContext();

// Wrapper component that creates global states and provides it to the context.
// The idea is that we wrap App in this, allowing the context to provide this data wherever we need.
// Note: children of this component will be passed as props.children.
export const GlobalProvider = (props) => {
    // This is the state we will use for storing user data.
    // You might also consider having game states here and passing to the context's provider below.
    const [user, setUser] = useState();

    return ( 
        <GlobalContext.Provider
            value={{ user, setUser }} // provide state variables to context.
        >
            {props.children}
        </GlobalContext.Provider>
    );
};