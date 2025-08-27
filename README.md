**Detailed network presentation and results from early stages of training in [Network architecture presentation.pdf](Network-architecture-presentation.pdf) file**

The "Lightweight Super-Resolution Enhanced Network for Object Detection in Remote Sensing" aim is to allow the recognition of trained targets in situations where image quality cannot be guaranteed due to tough operational conditions and hardware limitations.
This is made possible by combining 3 machine learning technologies:

- Super-Resolution: The network is trained to enhance image resolution and clarity by feeding into it pairs on low and high resolution images and adjusting weights and parameters accordingly.
- Object-Detection: The networks learns to identify patterns in the images indicating the presence of the kind of objects it was trained for and can classify objects in different categories.
- Attention mechanisms: This technique is fondamental for joining Super Resolution and Object Detection and is implemented with modules with which the networks learns to identify information-rich areas of the images to prioritize during detection.

These techniques where carefully tuned and merged in my network.

<img width="1745" height="980" alt="immagine" src="https://github.com/user-attachments/assets/4fd02001-cbec-419b-9845-b3530ddac794" />
Network architecture. Attention modules Hilighted in orange.

<img width="1549" height="585" alt="immagine" src="https://github.com/user-attachments/assets/de537256-eb3a-48e5-ba44-b9979f9d8f67" />
Detected Space debris and reconstructed image of real-life captured photo.

