# Workflow Release Process

Article Description: Approval processes ensure that everyone involved is informed and processes can proceed as intended.
Last Updated: 4. August 2021
Published: No
Suggested: No

In Poool it is possible to have incoming invoices, outgoing invoices, orders and offers released by certain persons.

## Set up release processes

In order to be able to apply the different release processes at all, the release processes must be defined first. To do this, navigate to `System → Release Processes`. Press the button `+ Release Process` to add a new release process.

The first step is to give the release process a name and choose whether the **process is active or not**. You can set values for the sharing process. This means that it is automatically applied to any calculation between these two values.

You can also decide to which area the approval process should apply.

In the next step, you can define the different stages of the release process. Click `+ Step` to add another step in the release process. You can decide whether the invoice needs to be released by only one or more people. For each of these steps, you can assign a title and a description.

You can also decide here whether it is still possible to change the data after this step or not.

In this step, the person or role responsible for the release of the document and a fallback is also selected. Any number of steps can be defined.

## Apply release process

As soon as the individual release processes are created and active in the background, they are applied directly to the corresponding calculations. In the upper left corner, a symbol appears indicating the release process with the corresponding color of the release status. If the icon is highlighted in yellow, this means that the release process is still open. As soon as the color turns green, the release process is complete.

In this step, another release process can also be added manually. To do this, click on the release process icon and select the release process you want to add from the drop-down menu.

## Share document

To share a document, navigate to `Projects → Shares`. This will open a list of all documents that are waiting for a release. Filter by the person in charge to get a list of invoices for which you are responsible.

You can either share the document directly by clicking on the checkbox or open the release process on the right with the icon.

You can see which release processes were applied to the invoice, or you can add further processes.

To be able to share the open process, open the share process. You will then see the information about the process, comment on it and close the process with the Share button.

You can also take a closer look at the document before sharing it. To do this, click on the document you want to open in the list.

You will enter the calculation of the document, where you can edit and adjust all positions. To be able to release your step, click on the share icon again and open the share mask by selecting the desired release process.

In the next step, you can write a comment as described above and add it with `+ Comment` . Click the button Share to release the step.

## Control Release

In the individual overview lists of the documents (e.g. outgoing invoices, offers, etc.) you can see the current status of an invoice and also follow the previous process. If you click on the release icon, you can see in which release step the document is or when and by whom the document was released. The release process can also be reset a step or deleted with the round arrow.

## Widget My releases

A “My releases” widget is available to all users in Poool. Employees can view and edit your pending releases directly. The widget can be added by double-clicking on the dashboard.

Important: If releases are requested for areas that are not accessed, sensitive information such as totals is hidden and the user is informed. In these cases, the workflow administrator should step in and grant the release or the user the appropriate permissions.

## Allowances

For more control over your releases, special permissions are available. In principle, all employees can grant releases. Selected employees can view all pending releases via special permissions (management) and, if necessary, edit them directly (administration). The permissions can be granted under `Users → Roles / Users → Administration`. Release administrators can grant all releases or reset release processes. We recommend that at least one person be designated as release administrator in order to be able to initiate stuck processes.