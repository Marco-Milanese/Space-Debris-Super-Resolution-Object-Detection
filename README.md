**Detailed network presentation and results from early stages of training in [Network-presentation.pdf](Network-presentation.pdf) file**

The "Lightweight Super-Resolution Enhanced Network for Object Detection in Remote Sensing" aim is to allow the recognition of trained targets in situations where image quality cannot be guaranteed due to tough operational conditions and hardware limitations.
This is made possible by combining 3 machine learning technologies:

- Super-Resolution: The network is trained to enhance image resolution and clarity by feeding into it pairs on low and high resolution images and adjusting weights and parameters accordingly.
- Object-Detection: The networks learns to identify patterns in the images indicating the presence of the kind of objects it was trained for and can classify objects in different categories.
- Attention mechanisms: This technique is fondamental for joining Super Resolution and Object Detection and is implemented with modules with which the networks learns to identify information-rich areas of the images to prioritize during detection.

These techniques where carefully tuned and merged in my network.

<img width="1753" height="1035" alt="Screenshot 2025-10-14 001037" src="https://github.com/user-attachments/assets/1d110522-922a-40f9-b962-94ce4d4d52d8" />

Network architecture. Attention modules Hilighted in orange.

<img width="1758" height="618" alt="Screenshot 2025-10-20 120846" src="https://github.com/user-attachments/assets/308e716b-704a-4765-979a-6280ec2df18f" />
Comparison between Low resolution image, network output and reference image.

The network performs excellently, obtaining near-perfect reconstruction and high detection accuracy, as shown in the testing data below:
<img width="1474" height="691" alt="Screenshot 2025-10-20 120907" src="https://github.com/user-attachments/assets/789457de-1968-43c8-88be-cb886aa7509b" />
Super Resolution performance
<img width="1349" height="801" alt="Screenshot 2025-10-20 120919" src="https://github.com/user-attachments/assets/de08549b-66e0-48fc-9ee9-6bd7ad86b194" />
Object Detection performance
